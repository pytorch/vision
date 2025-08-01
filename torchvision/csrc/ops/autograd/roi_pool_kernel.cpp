#include "../roi_pool.h"

#include <torch/autograd.h>
#include <torch/types.h>

#include <utility>

namespace vision {
namespace ops {

namespace {

class ROIPoolFunction : public torch::autograd::Function<ROIPoolFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& rois,
      double spatial_scale,
      const c10::SymInt& pooled_height,
      const c10::SymInt& pooled_width) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["input_shape"] = input.sym_sizes();
    at::AutoDispatchBelowADInplaceOrView g;
    auto result = roi_pool_symint(
        input, rois, spatial_scale, pooled_height, pooled_width);

    auto output = std::get<0>(result);
    auto argmax = std::get<1>(result);
    ctx->save_for_backward({rois, argmax});
    ctx->mark_non_differentiable({argmax});

    return {output, argmax};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto argmax = saved[1];
    auto input_shape = ctx->saved_data["input_shape"].toList();
    auto grad_in = detail::_roi_pool_backward_symint(
        grad_output[0],
        rois,
        argmax,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toSymInt(),
        ctx->saved_data["pooled_width"].toSymInt(),
        input_shape[0].get().toSymInt(),
        input_shape[1].get().toSymInt(),
        input_shape[2].get().toSymInt(),
        input_shape[3].get().toSymInt());

    return {
        grad_in,
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
      const torch::autograd::Variable& grad,
      const torch::autograd::Variable& rois,
      const torch::autograd::Variable& argmax,
      double spatial_scale,
      c10::SymInt pooled_height,
      c10::SymInt pooled_width,
      c10::SymInt batch_size,
      c10::SymInt channels,
      c10::SymInt height,
      c10::SymInt width) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto grad_in = detail::_roi_pool_backward_symint(
        grad,
        rois,
        argmax,
        spatial_scale,
        std::move(pooled_height),
        std::move(pooled_width),
        std::move(batch_size),
        std::move(channels),
        std::move(height),
        std::move(width));

    return {grad_in};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    TORCH_CHECK(0, "double backwards on roi_pool not supported");
  }
};

std::tuple<at::Tensor, at::Tensor> roi_pool_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width) {
  auto result = ROIPoolFunction::apply(
      input, rois, spatial_scale, pooled_height, pooled_width);

  return std::make_tuple(result[0], result[1]);
}

at::Tensor roi_pool_backward_autograd(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    c10::SymInt batch_size,
    c10::SymInt channels,
    c10::SymInt height,
    c10::SymInt width) {
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

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_pool"),
      TORCH_FN(roi_pool_autograd));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_pool_backward"),
      TORCH_FN(roi_pool_backward_autograd));
}

} // namespace ops
} // namespace vision
