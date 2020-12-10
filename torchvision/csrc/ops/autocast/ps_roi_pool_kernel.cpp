#include "../ps_roi_pool.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

std::tuple<at::Tensor, at::Tensor> ps_roi_pool_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto result = ps_roi_pool(
      at::autocast::cached_cast(at::kFloat, input),
      at::autocast::cached_cast(at::kFloat, rois),
      spatial_scale,
      pooled_height,
      pooled_width);

  return std::make_tuple(
      std::get<0>(result).to(input.scalar_type()),
      std::get<1>(result).to(input.scalar_type()));
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl("ps_roi_pool", ps_roi_pool_autocast);
}

} // namespace ops
} // namespace vision
