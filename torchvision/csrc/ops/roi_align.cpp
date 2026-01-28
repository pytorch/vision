#include "roi_align.h"

#include <torch/csrc/stable/library.h>

namespace vision {
namespace ops {

// Note: With stable ABI, ops are called directly via torch.ops.torchvision.*
// The dispatcher wrapper functions are no longer needed.

} // namespace ops
} // namespace vision

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "roi_align(Tensor input, Tensor rois, float spatial_scale, "
      "SymInt pooled_height, SymInt pooled_width, int sampling_ratio, "
      "bool aligned) -> Tensor");
  m.def(
      "_roi_align_backward(Tensor grad, Tensor rois, float spatial_scale, "
      "SymInt pooled_height, SymInt pooled_width, SymInt batch_size, "
      "SymInt channels, SymInt height, SymInt width, int sampling_ratio, "
      "bool aligned) -> Tensor");
}
