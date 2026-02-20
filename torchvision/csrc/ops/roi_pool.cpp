#include "roi_pool.h"

#include <torch/csrc/stable/library.h>

namespace vision {
namespace ops {

// Note: With stable ABI, ops are called directly via torch.ops.torchvision.*
// The dispatcher wrapper functions are no longer needed.

} // namespace ops
} // namespace vision

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "roi_pool(Tensor input, Tensor rois, float spatial_scale, "
      "SymInt pooled_height, SymInt pooled_width) -> (Tensor, Tensor)");
  m.def(
      "_roi_pool_backward(Tensor grad, Tensor rois, Tensor argmax, "
      "float spatial_scale, SymInt pooled_height, SymInt pooled_width, "
      "SymInt batch_size, SymInt channels, SymInt height, SymInt width) -> Tensor");
}
