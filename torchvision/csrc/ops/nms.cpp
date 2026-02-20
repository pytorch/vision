#include "nms.h"

#include <torch/csrc/stable/library.h>

namespace vision {
namespace ops {

// Note: With stable ABI, ops are called directly via torch.ops.torchvision.nms
// The dispatcher wrapper functions are no longer needed.

} // namespace ops
} // namespace vision

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor");
}
