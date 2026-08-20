#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <string>

#include "../StableABICompat.h"
#include "deform_conv2d_metal_shader.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

AOTIMetalShaderLibraryHandle deform_conv2d_shader_library() {
  static AOTIMetalShaderLibraryHandle library = []() {
    AOTIMetalShaderLibraryHandle handle = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_mps_create_shader_library(
        deform_conv2d_metal_shader, &handle));
    return handle;
  }();
  return library;
}

const char* metal_type_string(torch::headeronly::ScalarType scalar_type) {
  if (scalar_type == torch::headeronly::ScalarType::Float) {
    return "float";
  }
  if (scalar_type == torch::headeronly::ScalarType::Half) {
    return "half";
  }
  return "";
}

// The im2col kernel takes its (h, w)-style pairs as Metal `int2&` params (8
// bytes). The shim only sets int64 args, so pack each pair into one int64
// (little-endian: x in the low 32 bits, y in the high 32 bits). Single `int&`
// params read the low 32 bits of the int64, and `bool&` reads the low byte.
int64_t pack_int2(int64_t x, int64_t y) {
  return static_cast<int64_t>(
      static_cast<uint32_t>(x) | (static_cast<uint64_t>(static_cast<uint32_t>(y)) << 32));
}

struct DeformIm2colArgs {
  AtenTensorHandle input;
  AtenTensorHandle offset;
  AtenTensorHandle mask;
  AtenTensorHandle columns;
  int64_t input_size;
  int64_t weight_size;
  int64_t pad;
  int64_t stride;
  int64_t dilation;
  int64_t batch;
  int64_t in_channels;
  int64_t n_offset_grps;
  int64_t out_size;
  int64_t use_mask;
  uint64_t num_kernels;
};

void deform_im2col_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* a = static_cast<const DeformIm2colArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 0, a->input));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 1, a->offset));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 2, a->mask));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 3, a->input_size));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 4, a->weight_size));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 5, a->pad));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 6, a->stride));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 7, a->dilation));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 8, a->batch));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 9, a->in_channels));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 10, a->n_offset_grps));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 11, a->out_size));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 12, a->use_mask));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 13, a->columns));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(func, a->num_kernels));
}

Tensor deform_conv2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& mask,
    const Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask) {
  Tensor input_c = torch::stable::contiguous(input);
  Tensor weight_c = torch::stable::contiguous(weight);
  Tensor offset_c = torch::stable::contiguous(offset);
  Tensor mask_c = torch::stable::contiguous(mask);
  Tensor bias_c = torch::stable::contiguous(bias);

  STD_TORCH_CHECK(input_c.dim() == 4, "Input tensor must be 4D");
  STD_TORCH_CHECK(weight_c.dim() == 4, "Weight tensor must be 4D");
  STD_TORCH_CHECK(offset_c.dim() == 4, "Offset tensor must be 4D");
  STD_TORCH_CHECK(
      !use_mask || mask_c.dim() == 4,
      "Mask tensor must be 4D if use_mask is true");
  STD_TORCH_CHECK(
      input_c.device().type() == torch::headeronly::DeviceType::MPS,
      "input must be a MPS tensor");

  int64_t batch = input_c.size(0);
  int64_t in_channels = input_c.size(1);
  int64_t in_h = input_c.size(2);
  int64_t in_w = input_c.size(3);
  int64_t weight_h = weight_c.size(2);
  int64_t weight_w = weight_c.size(3);
  int64_t out_channels = weight_c.size(0);
  int64_t ker_h = dilation_h * (weight_h - 1) + 1;
  int64_t ker_w = dilation_w * (weight_w - 1) + 1;
  int64_t out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int64_t out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  STD_TORCH_CHECK(
      weight_c.size(1) * n_weight_grps == in_channels,
      "Input channels must equal weight.size(1) * n_weight_grps");
  STD_TORCH_CHECK(
      out_channels % n_weight_grps == 0,
      "Weight tensor's out channels must be divisible by n_weight_grps");
  STD_TORCH_CHECK(out_h > 0 && out_w > 0, "Calculated output size too small");

  Tensor columns = torch::stable::new_empty(
      input_c, {in_channels * weight_h * weight_w, batch * out_h * out_w});

  const std::string kernel = "deformable_im2col_" +
      std::string(metal_type_string(input_c.scalar_type()));
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(
      deform_conv2d_shader_library(), kernel.c_str(), &func));

  DeformIm2colArgs args{
      input_c.get(),
      offset_c.get(),
      // When use_mask is false the kernel never reads the mask buffer; bind the
      // input handle as a harmless placeholder so a valid buffer is always set.
      use_mask ? mask_c.get() : input_c.get(),
      columns.get(),
      pack_int2(in_h, in_w),
      pack_int2(weight_h, weight_w),
      pack_int2(pad_h, pad_w),
      pack_int2(stride_h, stride_w),
      pack_int2(dilation_h, dilation_w),
      batch,
      in_channels,
      n_offset_grps,
      pack_int2(out_h, out_w),
      use_mask ? 1 : 0,
      static_cast<uint64_t>(in_channels * out_h * out_w * batch)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_run_command_block(func, &deform_im2col_encode, &args));

  int64_t in_channels_per_grp = in_channels / n_weight_grps;
  int64_t out_channels_per_grp = out_channels / n_weight_grps;
  Tensor weight_grouped = torch::stable::view(
      weight_c,
      {n_weight_grps,
       out_channels_per_grp,
       in_channels_per_grp,
       weight_h,
       weight_w});
  Tensor columns_grouped = torch::stable::view(
      columns,
      {n_weight_grps,
       (in_channels * weight_h * weight_w) / n_weight_grps,
       batch * out_h * out_w});
  Tensor weight_reshaped = torch::stable::reshape(
      weight_grouped, {n_weight_grps, out_channels_per_grp, -1});
  Tensor out_grouped = torch::stable::matmul(weight_reshaped, columns_grouped);
  Tensor out = torch::stable::transpose(
      torch::stable::reshape(
          out_grouped,
          {n_weight_grps * out_channels_per_grp, batch, out_h, out_w}),
      0,
      1);
  Tensor bias_view =
      torch::stable::view(bias_c, {1, out_channels, 1, 1});
  // subtract(out, bias_view, alpha=-1) computes out - (-1) * bias_view, i.e.
  // out + bias_view; stable ops.h ships subtract but not add.
  return torch::stable::subtract(out, bias_view, /*alpha=*/-1.0);
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl("deform_conv2d", TORCH_BOX(&deform_conv2d_forward_kernel));
}

} // namespace ops
} // namespace vision
