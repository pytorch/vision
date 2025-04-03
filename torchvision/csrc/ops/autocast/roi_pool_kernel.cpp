#include "../roi_pool.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

template <c10::DispatchKey autocast_key, c10::DeviceType device_type>
std::tuple<at::Tensor, at::Tensor> roi_pool_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(autocast_key);
  auto result = roi_pool(
      at::autocast::cached_cast(at::kFloat, input, device_type),
      at::autocast::cached_cast(at::kFloat, rois, device_type),
      spatial_scale,
      pooled_height,
      pooled_width);

  return std::make_tuple(
      std::get<0>(result).to(input.scalar_type()),
      std::get<1>(result).to(input.scalar_type()));
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_pool"),
      TORCH_FN((roi_pool_autocast<
                c10::DispatchKey::Autocast,
                c10::DeviceType::CUDA>)));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_pool"),
      TORCH_FN((roi_pool_autocast<
                c10::DispatchKey::AutocastCPU,
                c10::DeviceType::CPU>)));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastXPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_pool"),
      TORCH_FN((roi_pool_autocast<
                c10::DispatchKey::AutocastXPU,
                c10::DeviceType::XPU>)));
}

} // namespace ops
} // namespace vision
