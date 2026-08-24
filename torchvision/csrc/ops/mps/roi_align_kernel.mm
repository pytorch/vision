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

struct RoiAlignForwardLaunchArgs {
  AtenTensorHandle input;
  AtenTensorHandle rois;
  AtenTensorHandle output;
  float spatial_scale;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t sampling_ratio;
  bool aligned;
  uint64_t output_size;
};

void roi_align_forward_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* launch_args =
      static_cast<const RoiAlignForwardLaunchArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  // [N, C, H, W]
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 0, launch_args->input));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 1, launch_args->rois));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 2, launch_args->output));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 3, &launch_args->spatial_scale, sizeof(float)));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 4, launch_args->channels));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 5, launch_args->height));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 6, launch_args->width));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 7, launch_args->pooled_height));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 8, launch_args->pooled_width));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 9, launch_args->sampling_ratio));
  TORCH_ERROR_CODE_CHECK(
      torch_mps_set_arg_bytes(func, 10, &launch_args->aligned, sizeof(bool)));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_dispatch_single(func, launch_args->output_size));
}

struct RoiAlignBackwardLaunchArgs {
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
  bool aligned;
  float spatial_scale;
  int64_t n_stride;
  int64_t c_stride;
  int64_t h_stride;
  int64_t w_stride;
};

void roi_align_backward_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* launch_args =
      static_cast<const RoiAlignBackwardLaunchArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  // [N, C, H, W]
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 0, launch_args->grad));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 1, launch_args->rois));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 2, launch_args->grad_input));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 3, launch_args->output_size));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 4, launch_args->channels));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 5, launch_args->height));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 6, launch_args->width));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 7, launch_args->pooled_height));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 8, launch_args->pooled_width));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 9, launch_args->sampling_ratio));
  TORCH_ERROR_CODE_CHECK(
      torch_mps_set_arg_bytes(func, 10, &launch_args->aligned, sizeof(bool)));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 11, &launch_args->spatial_scale, sizeof(float)));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 12, launch_args->n_stride));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 13, launch_args->c_stride));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 14, launch_args->h_stride));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 15, launch_args->w_stride));
  // One thread per pooled-output element. The kernel scatters into grad_input
  // with atomic_add for overlapping RoIs.
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(
      func, static_cast<uint64_t>(launch_args->output_size)));
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
  STD_TORCH_CHECK(
      input.get_device_index() == rois.get_device_index(),
      "input should be on the same device as rois");
  STD_TORCH_CHECK(
      input.scalar_type() == rois.scalar_type(),
      "input should have the same type as rois");

  int64_t num_rois = rois.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);

  Tensor output = torch::stable::new_zeros(
      input, {num_rois, channels, pooled_height, pooled_width});

  int64_t output_size = num_rois * pooled_height * pooled_width * channels;

  if (output.numel() == 0) {
    return output;
  }

  auto input_ = torch::stable::contiguous(input);
  auto rois_ = torch::stable::contiguous(rois);

  float spatial_scale_f = static_cast<float>(spatial_scale);

  const std::string kernel =
      "roi_align_" + std::string(mps::metal_type_string(input.scalar_type()));
  AOTIMetalKernelFunctionHandle func = mps::visionKernelFunction(kernel);

  RoiAlignForwardLaunchArgs launch_args{
      input_.get(),
      rois_.get(),
      output.get(),
      spatial_scale_f,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned,
      static_cast<uint64_t>(output_size)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &roi_align_forward_encode, &launch_args));
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
  STD_TORCH_CHECK(
      grad.get_device_index() == rois.get_device_index(),
      "grad should be on the same device as rois");
  STD_TORCH_CHECK(
      grad.scalar_type() == rois.scalar_type(),
      "grad should have the same type as rois");

  Tensor grad_input =
      torch::stable::new_zeros(grad, {batch_size, channels, height, width});

  if (grad.numel() == 0) {
    return grad_input;
  }

  int64_t n_stride = grad.stride(0);
  int64_t c_stride = grad.stride(1);
  int64_t h_stride = grad.stride(2);
  int64_t w_stride = grad.stride(3);
  int64_t output_size = grad.numel();

  auto rois_ = torch::stable::contiguous(rois);

  float spatial_scale_f = static_cast<float>(spatial_scale);

  const std::string kernel = "roi_align_backward_" +
      std::string(mps::metal_type_string(grad.scalar_type()));
  AOTIMetalKernelFunctionHandle func = mps::visionKernelFunction(kernel);

  RoiAlignBackwardLaunchArgs launch_args{
      grad.get(),
      rois_.get(),
      grad_input.get(),
      output_size,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned,
      spatial_scale_f,
      n_stride,
      c_stride,
      h_stride,
      w_stride};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &roi_align_backward_encode, &launch_args));
  return grad_input;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl("roi_align", TORCH_BOX(&roi_align_forward_kernel));
  m.impl("_roi_align_backward", TORCH_BOX(&roi_align_backward_kernel));
}

} // namespace ops
} // namespace vision
