#include "roi_align.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor roi_align(
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    int64_t pooled_height, // The height of the pooled feature map.
    int64_t pooled_width, // The width of the pooled feature
    int64_t sampling_ratio, // The number of points to sample in each bin
    bool aligned) // The flag for pixel shift
// along each axis.
{
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.roi_align.roi_align");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_align", "")
                       .typed<decltype(roi_align)>();
  return op.call(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

at::Tensor roi_align_symint(
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    c10::SymInt pooled_height, // The height of the pooled feature map.
    c10::SymInt pooled_width, // The width of the pooled feature
    int64_t sampling_ratio, // The number of points to sample in each bin
    bool aligned) // The flag for pixel shift
// along each axis.
{
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.roi_align.roi_align");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_align", "")
                       .typed<decltype(roi_align_symint)>();
  return op.call(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

namespace detail {

at::Tensor _roi_align_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_roi_align_backward", "")
          .typed<decltype(_roi_align_backward)>();
  return op.call(
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
      aligned);
}

at::Tensor _roi_align_backward_symint(
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
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_roi_align_backward", "")
          .typed<decltype(_roi_align_backward_symint)>();
  return op.call(
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
      aligned);
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, int sampling_ratio, bool aligned) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_roi_align_backward(Tensor grad, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, SymInt batch_size, SymInt channels, SymInt height, SymInt width, int sampling_ratio, bool aligned) -> Tensor"));
}

} // namespace ops
} // namespace vision
