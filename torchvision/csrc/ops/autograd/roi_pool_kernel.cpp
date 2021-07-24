#include "../roi_pool.h"

#include <torch/autograd.h>
#include <torch/types.h>

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
      int64_t pooled_height,
      int64_t pooled_width) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["input_shape"] = input.sizes();
    at::AutoDispatchBelowADInplaceOrView g;
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
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto argmax = saved[1];
    auto input_shape = ctx->saved_data["input_shape"].toIntList();
    auto grad_in = detail::_roi_pool_backward(
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

    return {
        grad_in,
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable()};
  }
};

std::tuple<at::Tensor, at::Tensor> roi_pool_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  auto result = ROIPoolFunction::apply(
      input, rois, spatial_scale, pooled_height, pooled_width);

  return std::make_tuple(result[0], result[1]);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_pool"),
      TORCH_FN(roi_pool_autograd));
}

} // namespace ops
} // namespace vision
