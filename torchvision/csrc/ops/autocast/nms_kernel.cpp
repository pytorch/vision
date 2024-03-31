#include "../nms.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

template <c10::DispatchKey autocast_key, c10::DeviceType device_type>
at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(autocast_key);

  return nms(
      at::autocast::cached_cast(at::kFloat, dets, device_type),
      at::autocast::cached_cast(at::kFloat, scores, device_type),
      iou_threshold);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::nms"),
      TORCH_FN(
          (nms_autocast<c10::DispatchKey::Autocast, c10::DeviceType::CUDA>)));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::nms"),
      TORCH_FN(
          (nms_autocast<c10::DispatchKey::AutocastCPU, c10::DeviceType::CPU>)));
}

} // namespace ops
} // namespace vision
