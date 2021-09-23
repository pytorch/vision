#include "ps_roi_align.h"

#include <torch/types.h>

namespace vision {
namespace ops {

std::tuple<at::Tensor, at::Tensor> ps_roi_align(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::ps_roi_align", "")
                       .typed<decltype(ps_roi_align)>();
  return op.call(
      input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

namespace detail {

at::Tensor _ps_roi_align_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_ps_roi_align_backward", "")
          .typed<decltype(_ps_roi_align_backward)>();
  return op.call(
      grad,
      rois,
      channel_mapping,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      batch_size,
      channels,
      height,
      width);
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::ps_roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_ps_roi_align_backward(Tensor grad, Tensor rois, Tensor channel_mapping, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, int batch_size, int channels, int height, int width) -> Tensor"));
}

} // namespace ops
} // namespace vision
