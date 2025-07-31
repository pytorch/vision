#include "../roi_align.h"

#include <torch/autograd.h>
#include <torch/types.h>

#include <utility>

namespace vision {
namespace ops {

namespace {

class ROIAlignFunction : public torch::autograd::Function<ROIAlignFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& rois,
      double spatial_scale,
      const c10::SymInt& pooled_height,
      const c10::SymInt& pooled_width,
      int64_t sampling_ratio,
      bool aligned) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["sampling_ratio"] = sampling_ratio;
    ctx->saved_data["aligned"] = aligned;
    ctx->saved_data["input_shape"] = input.sym_sizes();
    ctx->save_for_backward({rois});
    at::AutoDispatchBelowADInplaceOrView g;
    auto result = roi_align_symint(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned);
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto input_shape = ctx->saved_data["input_shape"].toList();
    auto grad_in = detail::_roi_align_backward_symint(
        grad_output[0],
        rois,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toSymInt(),
        ctx->saved_data["pooled_width"].toSymInt(),
        input_shape[0].get().toSymInt(),
        input_shape[1].get().toSymInt(),
        input_shape[2].get().toSymInt(),
        input_shape[3].get().toSymInt(),
        ctx->saved_data["sampling_ratio"].toInt(),
        ctx->saved_data["aligned"].toBool());
    return {
        grad_in,
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable()};
  }
};

// TODO: There should be an easier way to do this
class ROIAlignBackwardFunction
    : public torch::autograd::Function<ROIAlignBackwardFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& grad,
      const torch::autograd::Variable& rois,
      double spatial_scale,
      c10::SymInt pooled_height,
      c10::SymInt pooled_width,
      c10::SymInt batch_size,
      c10::SymInt channels,
      c10::SymInt height,
      c10::SymInt width,
      int64_t sampling_ratio,
      bool aligned) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto result = detail::_roi_align_backward_symint(
        grad,
        rois,
        spatial_scale,
        std::move(pooled_height),
        std::move(pooled_width),
        std::move(batch_size),
        std::move(channels),
        std::move(height),
        std::move(width),
        sampling_ratio,
        aligned);
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    TORCH_CHECK(0, "double backwards on roi_align not supported");
  }
};

at::Tensor roi_align_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  return ROIAlignFunction::apply(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned)[0];
}

at::Tensor roi_align_backward_autograd(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width,
    c10::SymInt batch_size,
    c10::SymInt channels,
    c10::SymInt height,
    c10::SymInt width,
    int64_t sampling_ratio,
    bool aligned) {
  return ROIAlignBackwardFunction::apply(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio,
      aligned)[0];
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN(roi_align_autograd));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_align_backward"),
      TORCH_FN(roi_align_backward_autograd));
}

} // namespace ops
} // namespace vision
