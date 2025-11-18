#include "../deform_attn.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

at::Tensor deform_attn_autocast(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const int64_t im2col_step) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return deform_attn(
             value,
             spatial_shapes,
             level_start_index,
             sampling_loc,
             attn_weight,
             im2col_step)
      .to(value.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_attn"),
      TORCH_FN(deform_attn_autocast));
}

} // namespace ops
} // namespace vision
