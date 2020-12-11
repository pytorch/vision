#include "nms.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::nms", "")
                       .typed<decltype(nms)>();
  return op.call(dets, scores, iou_threshold);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor");
}

} // namespace ops
} // namespace vision
