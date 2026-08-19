#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>

#include "roi_pool_metal_shader.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

constexpr int64_t threadsPerBlock = 512;

AOTIMetalShaderLibraryHandle roi_pool_shader_library() {
  static AOTIMetalShaderLibraryHandle library = []() {
    AOTIMetalShaderLibraryHandle handle = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_mps_create_shader_library(roi_pool_metal_shader, &handle));
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

struct RoiPoolForwardLaunchArgs {
  AtenTensorHandle input;
  AtenTensorHandle rois;
  AtenTensorHandle output;
  AtenTensorHandle argmax;
  int64_t output_size;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  float spatial_scale;
  uint64_t grid;
  uint64_t threadgroup;
};

void roi_pool_forward_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* launch_args =
      static_cast<const RoiPoolForwardLaunchArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  // [N, C, H, W]
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 0, launch_args->input));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 1, launch_args->rois));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 2, launch_args->output));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 3, launch_args->argmax));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 4, launch_args->output_size));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 5, launch_args->channels));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 6, launch_args->height));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 7, launch_args->width));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 8, launch_args->pooled_height));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 9, launch_args->pooled_width));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 10, &launch_args->spatial_scale, sizeof(float)));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single_with_group_size(
      func, launch_args->grid, launch_args->threadgroup));
}

struct RoiPoolBackwardLaunchArgs {
  AtenTensorHandle grad_output;
  AtenTensorHandle rois;
  AtenTensorHandle argmax;
  AtenTensorHandle grad_input;
  int64_t output_size;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  float spatial_scale;
  int64_t n_stride;
  int64_t c_stride;
  int64_t h_stride;
  int64_t w_stride;
};

void roi_pool_backward_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* launch_args =
      static_cast<const RoiPoolBackwardLaunchArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  // [N, C, H, W]
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 0, launch_args->grad_output));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 1, launch_args->rois));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 2, launch_args->argmax));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 3, launch_args->grad_input));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 4, launch_args->output_size));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 5, launch_args->channels));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 6, launch_args->height));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 7, launch_args->width));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 8, launch_args->pooled_height));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 9, launch_args->pooled_width));
  TORCH_ERROR_CODE_CHECK(torch_mps_set_arg_bytes(
      func, 10, &launch_args->spatial_scale, sizeof(float)));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 11, launch_args->n_stride));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 12, launch_args->c_stride));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 13, launch_args->h_stride));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 14, launch_args->w_stride));
  // One thread per pooled-output element. dispatchThreads guarantees each
  // index is handled exactly once; the kernel scatters into grad_input with
  // atomic_add for overlapping RoIs.
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(
      func, static_cast<uint64_t>(launch_args->output_size)));
}

std::tuple<Tensor, Tensor> roi_pool_forward_kernel(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
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
  Tensor argmax = torch::stable::new_zeros(
      input,
      {num_rois, channels, pooled_height, pooled_width},
      torch::headeronly::ScalarType::Long);

  int64_t output_size = num_rois * pooled_height * pooled_width * channels;

  if (output.numel() == 0) {
    return std::make_tuple(output, argmax);
  }

  auto input_ = torch::stable::contiguous(input);
  auto rois_ = torch::stable::contiguous(rois);

  float spatial_scale_f = static_cast<float>(spatial_scale);

  const std::string kernel =
      "roi_pool_" + std::string(metal_type_string(input.scalar_type()));
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(
      roi_pool_shader_library(), kernel.c_str(), &func));

  const int64_t threadgroups = std::min(
      (output_size + threadsPerBlock - 1) / threadsPerBlock,
      static_cast<int64_t>(4096));

  RoiPoolForwardLaunchArgs launch_args{
      input_.get(),
      rois_.get(),
      output.get(),
      argmax.get(),
      output_size,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale_f,
      static_cast<uint64_t>(threadgroups * threadsPerBlock),
      static_cast<uint64_t>(threadsPerBlock)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &roi_pool_forward_encode, &launch_args));
  return std::make_tuple(output, argmax);
}

Tensor roi_pool_backward_kernel(
    const Tensor& grad,
    const Tensor& rois,
    const Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  STD_TORCH_CHECK(
      grad.device().type() == torch::headeronly::DeviceType::MPS,
      "grad must be a MPS tensor");
  STD_TORCH_CHECK(
      rois.device().type() == torch::headeronly::DeviceType::MPS,
      "rois must be a MPS tensor");
  STD_TORCH_CHECK(
      grad.scalar_type() != torch::headeronly::ScalarType::Half,
      "MPS does not support roi_pool backward with float16 inputs.");
  STD_TORCH_CHECK(
      argmax.device().type() == torch::headeronly::DeviceType::MPS,
      "argmax must be a MPS tensor");
  STD_TORCH_CHECK(
      grad.get_device_index() == rois.get_device_index(),
      "grad should be on the same device as rois");
  STD_TORCH_CHECK(
      grad.get_device_index() == argmax.get_device_index(),
      "grad should be on the same device as argmax");
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

  auto argmax_ = torch::stable::contiguous(argmax);
  auto rois_ = torch::stable::contiguous(rois);

  float spatial_scale_f = static_cast<float>(spatial_scale);

  const std::string kernel =
      "roi_pool_backward_" + std::string(metal_type_string(grad.scalar_type()));
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(
      roi_pool_shader_library(), kernel.c_str(), &func));

  RoiPoolBackwardLaunchArgs launch_args{
      grad.get(),
      rois_.get(),
      argmax_.get(),
      grad_input.get(),
      output_size,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale_f,
      n_stride,
      c_stride,
      h_stride,
      w_stride};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &roi_pool_backward_encode, &launch_args));
  return grad_input;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl("roi_pool", TORCH_BOX(&roi_pool_forward_kernel));
  m.impl("_roi_pool_backward", TORCH_BOX(&roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision
