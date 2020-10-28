#pragma once

#include "cpu/vision_cpu.h"

#if defined(WITH_CUDA) || defined(WITH_HIP)
#include "autocast.h"
#endif

// TODO: put this stuff in torchvision namespace

std::tuple<at::Tensor, at::Tensor> roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_pool", "")
                       .typed<decltype(roi_pool)>();
  return op.call(input, rois, spatial_scale, pooled_height, pooled_width);
}

#if defined(WITH_CUDA) || defined(WITH_HIP)
std::tuple<at::Tensor, at::Tensor> ROIPool_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return roi_pool(
             at::autocast::cached_cast(at::kFloat, input),
             at::autocast::cached_cast(at::kFloat, rois),
             spatial_scale,
             pooled_height,
             pooled_width)
      .to(input.scalar_type());
}
#endif

at::Tensor _roi_pool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::_roi_pool_backward", "")
                       .typed<decltype(_roi_pool_backward)>();
  return op.call(
      grad,
      rois,
      argmax,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width);
}

class ROIPoolFunction : public torch::autograd::Function<ROIPoolFunction> {
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
        roi_pool(input, rois, spatial_scale, pooled_height, pooled_width);
    auto output = std::get<0>(result);
    auto argmax = std::get<1>(result);
    ctx->save_for_backward({rois, argmax});
    ctx->mark_non_differentiable({argmax});
    return {output, argmax};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto argmax = saved[1];
    auto input_shape = ctx->saved_data["input_shape"].toIntList();
    auto grad_in = _roi_pool_backward(
        grad_output[0],
        rois,
        argmax,
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

// TODO: There should be an easier way to do this
class ROIPoolBackwardFunction
    : public torch::autograd::Function<ROIPoolBackwardFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::Variable grad,
      torch::autograd::Variable rois,
      torch::autograd::Variable argmax,
      const double spatial_scale,
      const int64_t pooled_height,
      const int64_t pooled_width,
      const int64_t batch_size,
      const int64_t channels,
      const int64_t height,
      const int64_t width) {
    at::AutoNonVariableTypeMode g;
    auto grad_in = _roi_pool_backward(
        grad,
        rois,
        argmax,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width);

    return {grad_in};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    TORCH_CHECK(0, "double backwards on roi_pool not supported");
  }
};

std::tuple<at::Tensor, at::Tensor> ROIPool_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width) {
  auto result = ROIPoolFunction::apply(
      input, rois, spatial_scale, pooled_height, pooled_width);

  return std::make_tuple(result[0], result[1]);
}

at::Tensor ROIPool_backward_autograd(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width) {
  return ROIPoolBackwardFunction::apply(
      grad,
      rois,
      argmax,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width)[0];
}