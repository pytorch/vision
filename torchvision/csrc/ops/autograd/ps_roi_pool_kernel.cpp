#include "../ps_roi_pool.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

class PSROIPoolFunction : public torch::autograd::Function<PSROIPoolFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& rois,
      double spatial_scale,
      c10::SymInt pooled_height,
      c10::SymInt pooled_width) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["input_shape"] = input.sym_sizes();
    at::AutoDispatchBelowADInplaceOrView g;
    auto result = ps_roi_pool_symint(
        input, rois, spatial_scale, pooled_height, pooled_width);

    auto output = std::get<0>(result);
    auto channel_mapping = std::get<1>(result);
    ctx->save_for_backward({rois, channel_mapping});
    ctx->mark_non_differentiable({channel_mapping});

    return {output, channel_mapping};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto channel_mapping = saved[1];
    auto input_shape = ctx->saved_data["input_shape"].toList();
    auto grad_in = detail::_ps_roi_pool_backward_symint(
        grad_output[0],
        rois,
        channel_mapping,
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
class PSROIPoolBackwardFunction
    : public torch::autograd::Function<PSROIPoolBackwardFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& grad,
      const torch::autograd::Variable& rois,
      const torch::autograd::Variable& channel_mapping,
      double spatial_scale,
      c10::SymInt pooled_height,
      c10::SymInt pooled_width,
      c10::SymInt batch_size,
      c10::SymInt channels,
      c10::SymInt height,
      c10::SymInt width) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto grad_in = detail::_ps_roi_pool_backward_symint(
        grad,
        rois,
        channel_mapping,
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
      const torch::autograd::variable_list& grad_output) {
    TORCH_CHECK(0, "double backwards on ps_roi_pool not supported");
  }
};

std::tuple<at::Tensor, at::Tensor> ps_roi_pool_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width) {
  auto result = PSROIPoolFunction::apply(
      input, rois, spatial_scale, pooled_height, pooled_width);

  return std::make_tuple(result[0], result[1]);
}

at::Tensor ps_roi_pool_backward_autograd(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    c10::SymInt batch_size,
    c10::SymInt channels,
    c10::SymInt height,
    c10::SymInt width) {
  return PSROIPoolBackwardFunction::apply(
      grad,
      rois,
      channel_mapping,
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
      TORCH_SELECTIVE_NAME("torchvision::ps_roi_pool"),
      TORCH_FN(ps_roi_pool_autograd));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_ps_roi_pool_backward"),
      TORCH_FN(ps_roi_pool_backward_autograd));
}

} // namespace ops
} // namespace vision
