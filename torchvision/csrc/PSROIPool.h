#pragma once

#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

std::tuple<at::Tensor, at::Tensor> PSROIPool_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return PSROIPool_forward_cuda(
        input, rois, spatial_scale, pooled_height, pooled_width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return PSROIPool_forward_cpu(
      input, rois, spatial_scale, pooled_height, pooled_width);
}

at::Tensor PSROIPool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& mapping_channel,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return PSROIPool_backward_cuda(
        grad,
        rois,
        mapping_channel,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return PSROIPool_backward_cpu(
      grad,
      rois,
      mapping_channel,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width);
}

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class PSROIPoolFunction : public torch::autograd::Function<PSROIPoolFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input,
      Variable rois,
      const double spatial_scale,
      const int64_t pooled_height,
      const int64_t pooled_width) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["input_shape"] = input.sizes();
    auto result = PSROIPool_forward(
        input, rois, spatial_scale, pooled_height, pooled_width);
    auto output = std::get<0>(result);
    auto channel_mapping = std::get<1>(result);
    ctx->save_for_backward({rois, channel_mapping});
    ctx->mark_non_differentiable({channel_mapping});
    return {output, channel_mapping};
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto channel_mapping = saved[1];
    auto input_shape = ctx->saved_data["input_shape"].toIntList();
    auto grad_in = PSROIPool_backward(
        grad_output[0],
        rois,
        channel_mapping,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toInt(),
        ctx->saved_data["pooled_width"].toInt(),
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3]);
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

std::tuple<Tensor, Tensor> ps_roi_pool(
    const Tensor& input,
    const Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width) {
  auto result = PSROIPoolFunction::apply(
      input, rois, spatial_scale, pooled_height, pooled_width);
  return std::tuple<Tensor, Tensor>(result[0], result[1]);
}
