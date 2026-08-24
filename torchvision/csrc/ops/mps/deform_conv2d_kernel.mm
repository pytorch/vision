#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <string>

#include "mps_stable_kernels.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

// TODO(stable-abi): use torch::stable::add once the shim adds aten::add.Tensor.
Tensor add_tensors(const Tensor& self, const Tensor& other) {
  return torch::stable::subtract(self, other, -1.0);
}

struct DeformConv2dIm2colLaunchArgs {
  AtenTensorHandle input;
  AtenTensorHandle offset;
  AtenTensorHandle mask;
  AtenTensorHandle columns;
  uint32_t input_size[2];
  uint32_t weight_size[2];
  uint32_t pad[2];
  uint32_t stride[2];
  uint32_t dilation[2];
  uint32_t batch;
  uint32_t in_channels;
  uint32_t n_offset_grps;
  uint32_t out_size[2];
  bool use_mask;
  uint64_t num_kernels;
};

void deform_conv2d_im2col_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* launch_args =
      static_cast<const DeformConv2dIm2colLaunchArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 0, launch_args->input));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 1, launch_args->offset));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 2, launch_args->mask));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 3, launch_args->input_size, sizeof(launch_args->input_size)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 4, launch_args->weight_size, sizeof(launch_args->weight_size)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 5, launch_args->pad, sizeof(launch_args->pad)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 6, launch_args->stride, sizeof(launch_args->stride)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 7, launch_args->dilation, sizeof(launch_args->dilation)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 8, &launch_args->batch, sizeof(uint32_t)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 9, &launch_args->in_channels, sizeof(uint32_t)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 10, &launch_args->n_offset_grps, sizeof(uint32_t)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 11, launch_args->out_size, sizeof(launch_args->out_size)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 12, &launch_args->use_mask, sizeof(bool)));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 13, launch_args->columns));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_dispatch_single(func, launch_args->num_kernels));
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
      !use_mask || mask_c.dim() == 4, "Mask tensor must be 4D if use_mask is true");
  STD_TORCH_CHECK(
      input_c.device().type() == torch::headeronly::DeviceType::MPS,
      "input must be a MPS tensor");
  STD_TORCH_CHECK(
      weight.device().type() == torch::headeronly::DeviceType::MPS,
      "weight must be a MPS tensor");
  STD_TORCH_CHECK(
      offset.device().type() == torch::headeronly::DeviceType::MPS,
      "offset must be a MPS tensor");
  STD_TORCH_CHECK(
      mask.device().type() == torch::headeronly::DeviceType::MPS,
      "mask must be a MPS tensor");
  STD_TORCH_CHECK(
      bias.device().type() == torch::headeronly::DeviceType::MPS,
      "bias must be a MPS tensor");

  uint32_t batch = input_c.size(0);
  uint32_t in_channels = input_c.size(1);
  uint32_t in_h = input_c.size(2);
  uint32_t in_w = input_c.size(3);
  uint32_t weight_h = weight_c.size(2);
  uint32_t weight_w = weight_c.size(3);
  uint32_t out_channels = weight_c.size(0);
  uint32_t ker_h = dilation_h * (weight_h - 1) + 1;
  uint32_t ker_w = dilation_w * (weight_w - 1) + 1;
  uint32_t out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  uint32_t out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;
  uint32_t pad_h_u = static_cast<uint32_t>(pad_h);
  uint32_t pad_w_u = static_cast<uint32_t>(pad_w);
  uint32_t stride_h_u = static_cast<uint32_t>(stride_h);
  uint32_t stride_w_u = static_cast<uint32_t>(stride_w);
  uint32_t dilation_h_u = static_cast<uint32_t>(dilation_h);
  uint32_t dilation_w_u = static_cast<uint32_t>(dilation_w);

  STD_TORCH_CHECK(
      weight_c.size(1) * n_weight_grps == in_channels,
      "Input channels (", in_channels,
      ") must equal weight.size(1) * n_weight_grps (", weight_c.size(1),
      " * ", n_weight_grps, ")");
  STD_TORCH_CHECK(
      weight_c.size(0) % n_weight_grps == 0,
      "Weight tensor's out channels (", weight_c.size(0),
      ") must be divisible by n_weight_grps (", n_weight_grps, ")");
  STD_TORCH_CHECK(
      offset_c.size(1) == n_offset_grps * 2 * weight_h * weight_w,
      "Offset tensor shape[1] is invalid: got ", offset_c.size(1),
      ", expected ", n_offset_grps * 2 * weight_h * weight_w);
  STD_TORCH_CHECK(
      !use_mask || mask_c.size(1) == n_offset_grps * weight_h * weight_w,
      "Mask tensor shape[1] is invalid: got ", mask_c.size(1),
      ", expected ", n_offset_grps * weight_h * weight_w);
  STD_TORCH_CHECK(
      in_channels % n_offset_grps == 0,
      "Input tensor channels (", in_channels,
      ") must be divisible by n_offset_grps (", n_offset_grps, ")");
  STD_TORCH_CHECK(
      offset_c.size(0) == batch,
      "Offset tensor batch size (", offset_c.size(0),
      ") must match input tensor batch size (", batch, ")");
  STD_TORCH_CHECK(
      offset_c.size(2) == out_h && offset_c.size(3) == out_w,
      "Offset tensor spatial dimensions (", offset_c.size(2), ", ",
      offset_c.size(3),
      ") must match calculated output dimensions (", out_h, ", ", out_w, ")");
  STD_TORCH_CHECK(
      !use_mask || mask_c.size(0) == batch,
      "Mask tensor batch size (", mask_c.size(0),
      ") must match input tensor batch size (", batch, ")");
  STD_TORCH_CHECK(
      !use_mask || (mask_c.size(2) == out_h && mask_c.size(3) == out_w),
      "Mask tensor spatial dimensions (", mask_c.size(2), ", ", mask_c.size(3),
      ") must match calculated output dimensions (", out_h, ", ", out_w, ")");
  STD_TORCH_CHECK(
      out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ", out_h, " out_w: ", out_w);

  Tensor columns = torch::stable::new_empty(
      input_c,
      {static_cast<int64_t>(in_channels) * weight_h * weight_w,
       static_cast<int64_t>(batch) * out_h * out_w});

  const std::string kernel = "deformable_im2col_" +
      std::string(mps::metal_type_string(input.scalar_type()));
  AOTIMetalKernelFunctionHandle func = mps::visionKernelFunction(kernel);

  int64_t num_kernels =
      static_cast<int64_t>(in_channels) * out_h * out_w * batch;

  DeformConv2dIm2colLaunchArgs launch_args{
      input_c.get(),
      offset_c.get(),
      mask_c.get(),
      columns.get(),
      {in_h, in_w},
      {weight_h, weight_w},
      {pad_h_u, pad_w_u},
      {stride_h_u, stride_w_u},
      {dilation_h_u, dilation_w_u},
      batch,
      in_channels,
      static_cast<uint32_t>(n_offset_grps),
      {out_h, out_w},
      use_mask,
      static_cast<uint64_t>(num_kernels)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &deform_conv2d_im2col_encode, &launch_args));

  int64_t in_channels_per_grp = in_channels / n_weight_grps;
  int64_t out_channels_per_grp = out_channels / n_weight_grps;
  Tensor weight_grouped = torch::stable::view(
      weight_c,
      {n_weight_grps, out_channels_per_grp, in_channels_per_grp, weight_h,
       weight_w});
  Tensor columns_grouped = torch::stable::view(
      columns,
      {n_weight_grps,
       (static_cast<int64_t>(in_channels) * weight_h * weight_w) /
           n_weight_grps,
       static_cast<int64_t>(batch) * out_h * out_w});
  Tensor weight_reshaped = torch::stable::reshape(
      weight_grouped, {n_weight_grps, out_channels_per_grp, -1});
  Tensor out_grouped = torch::stable::matmul(weight_reshaped, columns_grouped);
  Tensor out = torch::stable::transpose(
      torch::stable::reshape(
          out_grouped,
          {n_weight_grps * out_channels_per_grp, batch, out_h, out_w}),
      0,
      1);
  Tensor bias_view = torch::stable::view(
      bias_c, {1, static_cast<int64_t>(out_channels), 1, 1});
  return add_tensors(out, bias_view);
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl("deform_conv2d", TORCH_BOX(&deform_conv2d_forward_kernel));
}

} // namespace ops
} // namespace vision
