#include "../roi_align.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

template <c10::DispatchKey autocast_key, c10::DeviceType device_type>
at::Tensor roi_align_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(autocast_key);
  return roi_align(
             at::autocast::cached_cast(at::kFloat, input, device_type),
             at::autocast::cached_cast(at::kFloat, rois, device_type),
             spatial_scale,
             pooled_height,
             pooled_width,
             sampling_ratio,
             aligned)
      .to(input.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN((roi_align_autocast<
                c10::DispatchKey::Autocast,
                c10::DeviceType::CUDA>)));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN((roi_align_autocast<
                c10::DispatchKey::AutocastCPU,
                c10::DeviceType::CPU>)));
}

} // namespace ops
} // namespace vision
