#include "ps_roi_pool.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

std::tuple<at::Tensor, at::Tensor> ps_roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.ps_roi_pool.ps_roi_pool");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::ps_roi_pool", "")
                       .typed<decltype(ps_roi_pool)>();
  return op.call(input, rois, spatial_scale, pooled_height, pooled_width);
}

std::tuple<at::Tensor, at::Tensor> ps_roi_pool_symint(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    c10::SymInt pooled_height,
    c10::SymInt pooled_width) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.ps_roi_pool.ps_roi_pool");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::ps_roi_pool", "")
                       .typed<decltype(ps_roi_pool_symint)>();
  return op.call(input, rois, spatial_scale, pooled_height, pooled_width);
}

namespace detail {

at::Tensor _ps_roi_pool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_ps_roi_pool_backward", "")
          .typed<decltype(_ps_roi_pool_backward)>();
  return op.call(
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
}

at::Tensor _ps_roi_pool_backward_symint(
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
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_ps_roi_pool_backward", "")
          .typed<decltype(_ps_roi_pool_backward_symint)>();
  return op.call(
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
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::ps_roi_pool(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_ps_roi_pool_backward(Tensor grad, Tensor rois, Tensor channel_mapping, float spatial_scale, SymInt pooled_height, SymInt pooled_width, SymInt batch_size, SymInt channels, SymInt height, SymInt width) -> Tensor"));
}

} // namespace ops
} // namespace vision
