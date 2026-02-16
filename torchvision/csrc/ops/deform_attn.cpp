#include "deform_attn.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor deform_attn(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const int64_t im2col_step) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.deform_attn.deform_attn");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::deform_attn", "")
                       .typed<decltype(deform_attn)>();
  return op.call(
      value,
      spatial_shapes,
      level_start_index,
      sampling_loc,
      attn_weight,
      im2col_step);
}

at::Tensor deform_attn_symint(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const c10::SymInt im2col_step) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.deform_attn.deform_attn");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::deform_attn", "")
                       .typed<decltype(deform_attn_symint)>();
  return op.call(
      value,
      spatial_shapes,
      level_start_index,
      sampling_loc,
      attn_weight,
      im2col_step);
}

namespace detail {

std::tuple<at::Tensor, at::Tensor, at::Tensor> _deform_attn_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    int64_t im2col_step) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_deform_attn_backward", "")
          .typed<decltype(_deform_attn_backward)>();
  return op.call(
      value,
      spatial_shapes,
      level_start_index,
      sampling_loc,
      attn_weight,
      grad_output,
      im2col_step);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _deform_attn_backward_symint(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    c10::SymInt im2col_step) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_deform_attn_backward", "")
          .typed<decltype(_deform_attn_backward_symint)>();
  return op.call(
      value,
      spatial_shapes,
      level_start_index,
      sampling_loc,
      attn_weight,
      grad_output,
      im2col_step);
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::deform_attn(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, SymInt im2col_step) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_deform_attn_backward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, Tensor grad_output, SymInt im2col_step) -> (Tensor, Tensor, Tensor)"));
}

} // namespace ops
} // namespace vision
