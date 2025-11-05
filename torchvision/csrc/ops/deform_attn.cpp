#include "deform_attn.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor deform_attn(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step
) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.deform_attn.deform_attn");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::deform_attention", "")
                       .typed<decltype(deform_attn)>();
  return op.call(
    value,
    spatial_shapes,
    level_start_index,
    sampling_loc,
    attn_weight,
    im2col_step
);
}

at::Tensor deform_attn_symint(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const c10::SymInt im2col_step
) {
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
    im2col_step
);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::deform_attn(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, SymInt stride_h, SymInt stride_w, SymInt pad_h, SymInt pad_w, SymInt dilation_h, SymInt dilation_w, SymInt groups, SymInt offset_groups, bool use_mask) -> Tensor"));
}

} // namespace ops
} // namespace vision
