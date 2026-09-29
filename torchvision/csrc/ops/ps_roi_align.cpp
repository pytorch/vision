#include "ps_roi_align.h"

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/headeronly/version.h>

#include <array>

namespace vision {
namespace ops {

using torch::stable::Tensor;

std::tuple<Tensor, Tensor> ps_roi_align(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio) {
  std::array<StableIValue, 6> stack{
      torch::stable::detail::from(input),
      torch::stable::detail::from(rois),
      torch::stable::detail::from(spatial_scale),
      torch::stable::detail::from(pooled_height),
      torch::stable::detail::from(pooled_width),
      torch::stable::detail::from(sampling_ratio)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::ps_roi_align", "", stack.data(), TORCH_ABI_VERSION));
  return std::make_tuple(
      torch::stable::detail::to<Tensor>(stack[0]),
      torch::stable::detail::to<Tensor>(stack[1]));
}

namespace detail {

Tensor _ps_roi_align_backward(
    const Tensor& grad,
    const Tensor& rois,
    const Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  std::array<StableIValue, 11> stack{
      torch::stable::detail::from(grad),
      torch::stable::detail::from(rois),
      torch::stable::detail::from(channel_mapping),
      torch::stable::detail::from(spatial_scale),
      torch::stable::detail::from(pooled_height),
      torch::stable::detail::from(pooled_width),
      torch::stable::detail::from(sampling_ratio),
      torch::stable::detail::from(batch_size),
      torch::stable::detail::from(channels),
      torch::stable::detail::from(height),
      torch::stable::detail::from(width)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::_ps_roi_align_backward",
      "",
      stack.data(),
      TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

} // namespace detail

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "ps_roi_align(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, int sampling_ratio) -> (Tensor, Tensor)");
  m.def(
      "_ps_roi_align_backward(Tensor grad, Tensor rois, Tensor channel_mapping, float spatial_scale, SymInt pooled_height, SymInt pooled_width, int sampling_ratio, SymInt batch_size, SymInt channels, SymInt height, SymInt width) -> Tensor");
}

} // namespace ops
} // namespace vision
