#include "../nms.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return nms(
      at::autocast::cached_cast(at::kFloat, dets),
      at::autocast::cached_cast(at::kFloat, scores),
      iou_threshold);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_autocast));
}

} // namespace ops
} // namespace vision
