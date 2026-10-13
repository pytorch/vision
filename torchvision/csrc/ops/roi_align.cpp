#include "roi_align.h"

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/headeronly/version.h>

#include <array>

namespace vision {
namespace ops {

using torch::stable::Tensor;

Tensor roi_align(
    const Tensor& input, // Input feature map.
    const Tensor& rois, // List of ROIs to pool over.
    double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    int64_t pooled_height, // The height of the pooled feature map.
    int64_t pooled_width, // The width of the pooled feature
    int64_t sampling_ratio, // The number of points to sample in each bin
    bool aligned) // The flag for pixel shift
// along each axis.
{
  std::array<StableIValue, 7> stack{
      torch::stable::detail::from(input),
      torch::stable::detail::from(rois),
      torch::stable::detail::from(spatial_scale),
      torch::stable::detail::from(pooled_height),
      torch::stable::detail::from(pooled_width),
      torch::stable::detail::from(sampling_ratio),
      torch::stable::detail::from(aligned)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::roi_align", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

namespace detail {

Tensor _roi_align_backward(
    const Tensor& grad,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  std::array<StableIValue, 11> stack{
      torch::stable::detail::from(grad),
      torch::stable::detail::from(rois),
      torch::stable::detail::from(spatial_scale),
      torch::stable::detail::from(pooled_height),
      torch::stable::detail::from(pooled_width),
      torch::stable::detail::from(batch_size),
      torch::stable::detail::from(channels),
      torch::stable::detail::from(height),
      torch::stable::detail::from(width),
      torch::stable::detail::from(sampling_ratio),
      torch::stable::detail::from(aligned)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::_roi_align_backward", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

} // namespace detail

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "roi_align(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, int sampling_ratio, bool aligned) -> Tensor");
  m.def(
      "_roi_align_backward(Tensor grad, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, SymInt batch_size, SymInt channels, SymInt height, SymInt width, int sampling_ratio, bool aligned) -> Tensor");
}

} // namespace ops
} // namespace vision
