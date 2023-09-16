#include "nms.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.nms.nms");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::nms", "")
                       .typed<decltype(nms)>();
  return op.call(dets, scores, iou_threshold);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"));
}

} // namespace ops
} // namespace vision
