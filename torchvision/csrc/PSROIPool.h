#pragma once

#include "cpu/vision_cpu.h"

#if defined(WITH_CUDA) || defined(WITH_HIP)
#include "autocast.h"
#endif

// TODO: put this stuff in torchvision namespace

std::tuple<at::Tensor, at::Tensor> ps_roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::ps_roi_pool", "")
      .typed<decltype(ps_roi_pool)>();
  return op.call(input, rois, spatial_scale, pooled_height, pooled_width);
}

#if defined(WITH_CUDA) || defined(WITH_HIP)
std::tuple<at::Tensor, at::Tensor> PSROIPool_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto result = ps_roi_pool(
      at::autocast::cached_cast(at::kFloat, input),
      at::autocast::cached_cast(at::kFloat, rois),
      spatial_scale,
      pooled_height,
      pooled_width);

  return std::make_tuple(
      std::get<0>(result).to(input.scalar_type()),
      std::get<1>(result).to(input.scalar_type()));
}
#endif

at::Tensor _ps_roi_pool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& mapping_channel,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::_ps_roi_pool_backward", "")
      .typed<decltype(_ps_roi_pool_backward)>();
  return op.call(
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

class PSROIPoolFunction : public torch::autograd::Function<PSROIPoolFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::Variable input,
      torch::autograd::Variable rois,
      const double spatial_scale,
      const int64_t pooled_height,
      const int64_t pooled_width) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["input_shape"] = input.sizes();
    at::AutoNonVariableTypeMode g;
    auto result =
        ps_roi_pool(input, rois, spatial_scale, pooled_height, pooled_width);
    auto output = std::get<0>(result);
    auto mapping_channel = std::get<1>(result);
    ctx->save_for_backward({rois, mapping_channel});
    ctx->mark_non_differentiable({mapping_channel});
    return {output, mapping_channel};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto mapping_channel = saved[1];
    auto input_shape = ctx->saved_data["input_shape"].toIntList();
    auto grad_in = _ps_roi_pool_backward(
        grad_output[0],
        rois,
        mapping_channel,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toInt(),
        ctx->saved_data["pooled_width"].toInt(),
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3]);
    return {grad_in,
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()};
  }
};

