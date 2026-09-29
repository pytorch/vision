#include "nms.h"

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/headeronly/version.h>

#include <array>

namespace vision {
namespace ops {

using torch::stable::Tensor;

Tensor nms(const Tensor& dets, const Tensor& scores, double iou_threshold) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(dets),
      torch::stable::detail::from(scores),
      torch::stable::detail::from(iou_threshold)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::nms", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor");
}

} // namespace ops
} // namespace vision
