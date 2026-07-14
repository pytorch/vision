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
#include "roi_align_metal_shader.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

// Lazily compile the roi_align Metal shader library once for the process,
// mirroring the lazy-singleton the AOTInductor MPS backend generates. The
// handle lives for the process lifetime.
AOTIMetalShaderLibraryHandle roi_align_shader_library() {
  static AOTIMetalShaderLibraryHandle library = []() {
    AOTIMetalShaderLibraryHandle handle = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_mps_create_shader_library(roi_align_metal_shader, &handle));
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

// The MPS shim has no scalar-float arg setter, so spatial_scale rides in as a
// 1-element float32 tensor (same workaround as nms's iou_threshold).
// TODO(stable-abi): bind a float directly once aoti_torch_mps_set_arg_double /
// set_arg_bytes lands upstream (pytorch/pytorch).
struct RoiAlignForwardArgs {
  AtenTensorHandle input;
  AtenTensorHandle rois;
  AtenTensorHandle output;
  AtenTensorHandle spatial_scale;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t sampling_ratio;
  int64_t aligned;
  uint64_t output_size;
};

void roi_align_forward_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* a = static_cast<const RoiAlignForwardArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 0, a->input));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 1, a->rois));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 2, a->output));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 3, a->spatial_scale));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 4, a->channels));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 5, a->height));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 6, a->width));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 7, a->pooled_height));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 8, a->pooled_width));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 9, a->sampling_ratio));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 10, a->aligned));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(func, a->output_size));
}

struct RoiAlignBackwardArgs {
  AtenTensorHandle grad;
  AtenTensorHandle rois;
  AtenTensorHandle grad_input;
  int64_t output_size;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t sampling_ratio;
  int64_t aligned;
  AtenTensorHandle spatial_scale;
  int64_t n_stride;
  int64_t c_stride;
  int64_t h_stride;
  int64_t w_stride;
};

void roi_align_backward_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* a = static_cast<const RoiAlignBackwardArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 0, a->grad));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 1, a->rois));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 2, a->grad_input));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 3, a->output_size));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 4, a->channels));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 5, a->height));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 6, a->width));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 7, a->pooled_height));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 8, a->pooled_width));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 9, a->sampling_ratio));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 10, a->aligned));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 11, a->spatial_scale));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 12, a->n_stride));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 13, a->c_stride));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 14, a->h_stride));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 15, a->w_stride));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(
      func, static_cast<uint64_t>(a->output_size)));
}

Tensor make_scalar_tensor(const Tensor& ref, double value) {
  Tensor t = torch::stable::new_empty(
      ref, {1}, torch::headeronly::ScalarType::Float);
  torch::stable::fill_(t, value);
  return t;
}

Tensor roi_align_forward_kernel(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  STD_TORCH_CHECK(
      input.device().type() == torch::headeronly::DeviceType::MPS,
      "input must be a MPS tensor");
  STD_TORCH_CHECK(
      rois.device().type() == torch::headeronly::DeviceType::MPS,
      "rois must be a MPS tensor");
  STD_TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  int64_t num_rois = rois.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);

  Tensor output = torch::stable::new_empty(
      input, {num_rois, channels, pooled_height, pooled_width});
  int64_t output_size = output.numel();
  if (output_size == 0) {
    return output;
  }

  Tensor input_ = torch::stable::contiguous(input);
  Tensor rois_ = torch::stable::contiguous(rois);
  Tensor spatial_scale_t = make_scalar_tensor(input_, spatial_scale);

  const std::string kernel =
      "roi_align_" + std::string(metal_type_string(input_.scalar_type()));
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(
      roi_align_shader_library(), kernel.c_str(), &func));

  RoiAlignForwardArgs args{
      input_.get(),
      rois_.get(),
      output.get(),
      spatial_scale_t.get(),
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned ? 1 : 0,
      static_cast<uint64_t>(output_size)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &roi_align_forward_encode, &args));
  return output;
}

Tensor roi_align_backward_kernel(
    const Tensor& grad,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  STD_TORCH_CHECK(
      grad.device().type() == torch::headeronly::DeviceType::MPS,
      "grad must be a MPS tensor");
  STD_TORCH_CHECK(
      rois.device().type() == torch::headeronly::DeviceType::MPS,
      "rois must be a MPS tensor");
  STD_TORCH_CHECK(
      grad.scalar_type() != torch::headeronly::ScalarType::Half,
      "MPS does not support roi_align backward with float16 inputs.");

  Tensor grad_input = torch::stable::new_zeros(
      grad, {batch_size, channels, height, width});
  if (grad.numel() == 0) {
    return grad_input;
  }

  // Index the gradient with contiguous strides (computed from the pooled output
  // shape), so making grad contiguous lets us drop the runtime .stride() reads.
  Tensor grad_ = torch::stable::contiguous(grad);
  Tensor rois_ = torch::stable::contiguous(rois);
  Tensor spatial_scale_t = make_scalar_tensor(grad_, spatial_scale);

  int64_t output_size = grad_.numel();
  int64_t w_stride = 1;
  int64_t h_stride = pooled_width;
  int64_t c_stride = pooled_height * pooled_width;
  int64_t n_stride = channels * pooled_height * pooled_width;

  const std::string kernel = "roi_align_backward_" +
      std::string(metal_type_string(grad_.scalar_type()));
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(
      roi_align_shader_library(), kernel.c_str(), &func));

  RoiAlignBackwardArgs args{
      grad_.get(),
      rois_.get(),
      grad_input.get(),
      output_size,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned ? 1 : 0,
      spatial_scale_t.get(),
      n_stride,
      c_stride,
      h_stride,
      w_stride};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &roi_align_backward_encode, &args));
  return grad_input;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl("roi_align", TORCH_BOX(&roi_align_forward_kernel));
  m.impl("_roi_align_backward", TORCH_BOX(&roi_align_backward_kernel));
}

} // namespace ops
} // namespace vision
