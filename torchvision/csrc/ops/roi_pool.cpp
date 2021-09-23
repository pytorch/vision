#include "roi_pool.h"

#include <torch/types.h>

namespace vision {
namespace ops {

std::tuple<at::Tensor, at::Tensor> roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_pool", "")
                       .typed<decltype(roi_pool)>();
  return op.call(input, rois, spatial_scale, pooled_height, pooled_width);
}

namespace detail {

at::Tensor _roi_pool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
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

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_roi_pool_backward(Tensor grad, Tensor rois, Tensor argmax, float spatial_scale, int pooled_height, int pooled_width, int batch_size, int channels, int height, int width) -> Tensor"));
}

} // namespace ops
} // namespace vision
