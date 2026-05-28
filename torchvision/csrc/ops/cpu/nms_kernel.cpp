#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>

namespace vision {
namespace ops {

namespace {

namespace stable = torch::stable;
namespace headeronly = torch::headeronly;
using headeronly::ScalarType;
using stable::Tensor;

inline std::pair<Tensor, Tensor> dispatch_sort(
    const Tensor& self,
    bool is_stable,
    int64_t dim,
    bool descending) {
  constexpr int num_args = 4;
  std::array<StableIValue, num_args> stack{
      stable::detail::from(self),
      stable::detail::from(std::optional<bool>(is_stable)),
      stable::detail::from(dim),
      stable::detail::from(descending)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sort", "stable", stack.data(), TORCH_ABI_VERSION));
  return {
      stable::detail::to<Tensor>(stack[0]),
      stable::detail::to<Tensor>(stack[1])};
}

inline Tensor dispatch_mul(const Tensor& self, const Tensor& other) {
  constexpr int num_args = 2;
  std::array<StableIValue, num_args> stack{
      stable::detail::from(self), stable::detail::from(other)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::mul", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return stable::detail::to<Tensor>(stack[0]);
}

template <typename scalar_t>
Tensor nms_kernel_impl(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {
  STD_TORCH_CHECK(dets.is_cpu(), "dets must be a CPU tensor");
  STD_TORCH_CHECK(scores.is_cpu(), "scores must be a CPU tensor");
  STD_TORCH_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return stable::new_empty(dets, {0}, ScalarType::Long);
  }

  auto x1_t = stable::contiguous(stable::select(dets, 1, 0));
  auto y1_t = stable::contiguous(stable::select(dets, 1, 1));
  auto x2_t = stable::contiguous(stable::select(dets, 1, 2));
  auto y2_t = stable::contiguous(stable::select(dets, 1, 3));

  auto areas_t =
      dispatch_mul(stable::subtract(x2_t, x1_t), stable::subtract(y2_t, y1_t));

  auto [values, order_t] = dispatch_sort(scores, true, 0, true);

  auto ndets = dets.size(0);
  auto suppressed_t = stable::new_zeros(dets, {ndets}, ScalarType::Byte);
  auto keep_t = stable::new_zeros(dets, {ndets}, ScalarType::Long);

  auto suppressed = suppressed_t.mutable_data_ptr<uint8_t>();
  auto keep = keep_t.mutable_data_ptr<int64_t>();
  auto order = order_t.const_data_ptr<int64_t>();
  auto x1 = x1_t.const_data_ptr<scalar_t>();
  auto y1 = y1_t.const_data_ptr<scalar_t>();
  auto x2 = x2_t.const_data_ptr<scalar_t>();
  auto y2 = y2_t.const_data_ptr<scalar_t>();
  auto areas = areas_t.const_data_ptr<scalar_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }
    keep[num_to_keep++] = i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
        continue;
      }
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  return stable::narrow(keep_t, 0, 0, num_to_keep);
}

Tensor nms_kernel(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {
  STD_TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  STD_TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  STD_TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  STD_TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  auto result = stable::new_empty(dets, {0});

  THO_DISPATCH_SWITCH(
      dets.scalar_type(),
      "nms_kernel",
      THO_DISPATCH_CASE(ScalarType::Float, [&] {
        result = nms_kernel_impl<scalar_t>(dets, scores, iou_threshold);
      }) THO_DISPATCH_CASE(ScalarType::Double, [&] {
        result = nms_kernel_impl<scalar_t>(dets, scores, iou_threshold);
      }));
  return result;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("nms", TORCH_BOX(&nms_kernel));
}

} // namespace ops
} // namespace vision
