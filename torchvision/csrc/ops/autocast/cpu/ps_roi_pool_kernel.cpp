#include "../../ps_roi_pool.h"

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
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastCPU);
  auto result = ps_roi_pool(
      at::autocast::cached_cast(at::kFloat, input, c10::DeviceType::CPU),
      at::autocast::cached_cast(at::kFloat, rois, c10::DeviceType::CPU),
      spatial_scale,
      pooled_height,
      pooled_width);

  return std::make_tuple(
      std::get<0>(result).to(input.scalar_type()),
      std::get<1>(result).to(input.scalar_type()));
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, AutocastCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::ps_roi_pool"),
      TORCH_FN(ps_roi_pool_autocast));
}

} // namespace ops
} // namespace vision
