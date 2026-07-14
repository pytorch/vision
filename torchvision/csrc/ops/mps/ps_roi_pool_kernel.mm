#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "../StableABICompat.h"
#include "ps_roi_pool_metal_shader.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

AOTIMetalShaderLibraryHandle ps_roi_pool_shader_library() {
  static AOTIMetalShaderLibraryHandle library = []() {
    AOTIMetalShaderLibraryHandle handle = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_mps_create_shader_library(
        ps_roi_pool_metal_shader, &handle));
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

// spatial_scale rides in as a 1-element float32 tensor: the MPS shim has no
// scalar-float arg setter yet (same workaround as nms's iou_threshold).
Tensor make_scalar_tensor(const Tensor& ref, double value) {
  Tensor t = torch::stable::new_empty(
      ref, {1}, torch::headeronly::ScalarType::Float);
  torch::stable::fill_(t, value);
  return t;
}

struct PsRoiPoolForwardArgs {
  AtenTensorHandle input;
  AtenTensorHandle rois;
  AtenTensorHandle output;
  AtenTensorHandle channel_mapping;
  int64_t output_size;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t channels_out;
  AtenTensorHandle spatial_scale;
};

void ps_roi_pool_forward_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* a = static_cast<const PsRoiPoolForwardArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 0, a->input));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 1, a->rois));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 2, a->output));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 3, a->channel_mapping));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 4, a->output_size));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 5, a->channels));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 6, a->height));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 7, a->width));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 8, a->pooled_height));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 9, a->pooled_width));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 10, a->channels_out));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 11, a->spatial_scale));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(
      func, static_cast<uint64_t>(a->output_size)));
}

struct PsRoiPoolBackwardArgs {
  AtenTensorHandle grad;
  AtenTensorHandle rois;
  AtenTensorHandle channel_mapping;
  AtenTensorHandle grad_input;
  int64_t output_size;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t channels_out;
  AtenTensorHandle spatial_scale;
};

void ps_roi_pool_backward_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  const auto* a = static_cast<const PsRoiPoolBackwardArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 0, a->grad));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 1, a->rois));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 2, a->channel_mapping));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 3, a->grad_input));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 4, a->output_size));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 5, a->channels));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 6, a->height));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 7, a->width));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 8, a->pooled_height));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 9, a->pooled_width));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_int(func, 10, a->channels_out));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 11, a->spatial_scale));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(
      func, static_cast<uint64_t>(a->output_size)));
}

std::tuple<Tensor, Tensor> ps_roi_pool_forward_kernel(
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

  int64_t num_rois = rois.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  STD_TORCH_CHECK(
      channels % (pooled_height * pooled_width) == 0,
      "input channels must be a multiple of pooling height * pooling width");
  int64_t channels_out = channels / (pooled_height * pooled_width);

  Tensor output = torch::stable::new_empty(
      input, {num_rois, channels_out, pooled_height, pooled_width});
  Tensor channel_mapping = torch::stable::new_empty(
      input,
      {num_rois, channels_out, pooled_height, pooled_width},
      torch::headeronly::ScalarType::Long);
  int64_t output_size = output.numel();
  if (output_size == 0) {
    return std::make_tuple(output, channel_mapping);
  }

  Tensor input_ = torch::stable::contiguous(input);
  Tensor rois_ = torch::stable::contiguous(rois);
  Tensor spatial_scale_t = make_scalar_tensor(input_, spatial_scale);

  const std::string kernel =
      "ps_roi_pool_" + std::string(metal_type_string(input_.scalar_type()));
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(
      ps_roi_pool_shader_library(), kernel.c_str(), &func));

  PsRoiPoolForwardArgs args{
      input_.get(),
      rois_.get(),
      output.get(),
      channel_mapping.get(),
      output_size,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      channels_out,
      spatial_scale_t.get()};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &ps_roi_pool_forward_encode, &args));
  return std::make_tuple(output, channel_mapping);
}

Tensor ps_roi_pool_backward_kernel(
    const Tensor& grad,
    const Tensor& rois,
    const Tensor& channel_mapping,
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
      channel_mapping.device().type() == torch::headeronly::DeviceType::MPS,
      "channel_mapping must be a MPS tensor");
  STD_TORCH_CHECK(
      grad.scalar_type() != torch::headeronly::ScalarType::Half,
      "MPS does not support ps_roi_pool backward with float16 inputs.");

  Tensor grad_input = torch::stable::new_zeros(
      grad, {batch_size, channels, height, width});
  if (grad.numel() == 0) {
    return grad_input;
  }

  int64_t channels_out = channels / (pooled_height * pooled_width);

  Tensor grad_ = torch::stable::contiguous(grad);
  Tensor rois_ = torch::stable::contiguous(rois);
  Tensor channel_mapping_ = torch::stable::contiguous(channel_mapping);
  Tensor spatial_scale_t = make_scalar_tensor(grad_, spatial_scale);

  int64_t output_size = grad_.numel();

  const std::string kernel = "ps_roi_pool_backward_" +
      std::string(metal_type_string(grad_.scalar_type()));
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(
      ps_roi_pool_shader_library(), kernel.c_str(), &func));

  PsRoiPoolBackwardArgs args{
      grad_.get(),
      rois_.get(),
      channel_mapping_.get(),
      grad_input.get(),
      output_size,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      channels_out,
      spatial_scale_t.get()};
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, &ps_roi_pool_backward_encode, &args));
  return grad_input;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl("ps_roi_pool", TORCH_BOX(&ps_roi_pool_forward_kernel));
  m.impl("_ps_roi_pool_backward", TORCH_BOX(&ps_roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision
