#include "../../StableABICompat.h"
#include <torch/csrc/stable/library.h>

namespace vision {
namespace ops {

namespace {

using namespace vision::stable;

template <typename scalar_t>
Tensor nms_kernel_impl(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {
  VISION_CHECK(dets.is_cpu(), "dets must be a CPU tensor");
  VISION_CHECK(scores.is_cpu(), "scores must be a CPU tensor");
  VISION_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return empty({0}, kLong, Device(kCPU));
  }

  auto x1_t = torch::stable::contiguous(torch::stable::select(dets, 1, 0));
  auto y1_t = torch::stable::contiguous(torch::stable::select(dets, 1, 1));
  auto x2_t = torch::stable::contiguous(torch::stable::select(dets, 1, 2));
  auto y2_t = torch::stable::contiguous(torch::stable::select(dets, 1, 3));

  // Compute areas: (x2 - x1) * (y2 - y1)
  // Need to do this manually with data pointers
  auto ndets = dets.size(0);
  Tensor areas_t = empty({ndets}, dets.scalar_type(), Device(kCPU));

  auto x1_ptr = x1_t.const_data_ptr<scalar_t>();
  auto y1_ptr = y1_t.const_data_ptr<scalar_t>();
  auto x2_ptr = x2_t.const_data_ptr<scalar_t>();
  auto y2_ptr = y2_t.const_data_ptr<scalar_t>();
  auto areas_ptr = areas_t.mutable_data_ptr<scalar_t>();

  for (int64_t i = 0; i < ndets; i++) {
    areas_ptr[i] = (x2_ptr[i] - x1_ptr[i]) * (y2_ptr[i] - y1_ptr[i]);
  }

  // Sort scores descending
  auto [sorted_scores, order_t] = sort(scores, /*dim=*/0, /*descending=*/true);

  Tensor suppressed_t = zeros({ndets}, kByte, Device(kCPU));
  Tensor keep_t = zeros({ndets}, kLong, Device(kCPU));

  auto suppressed = suppressed_t.mutable_data_ptr<uint8_t>();
  auto keep = keep_t.mutable_data_ptr<int64_t>();
  auto order = order_t.const_data_ptr<int64_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }
    keep[num_to_keep++] = i;
    auto ix1 = x1_ptr[i];
    auto iy1 = y1_ptr[i];
    auto ix2 = x2_ptr[i];
    auto iy2 = y2_ptr[i];
    auto iarea = areas_ptr[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
        continue;
      }
      auto xx1 = std::max(ix1, x1_ptr[j]);
      auto yy1 = std::max(iy1, y1_ptr[j]);
      auto xx2 = std::min(ix2, x2_ptr[j]);
      auto yy2 = std::min(iy2, y2_ptr[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas_ptr[j] - inter);
      if (ovr > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  return torch::stable::narrow(keep_t, /*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

Tensor nms_kernel(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {
  VISION_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  VISION_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  VISION_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  VISION_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  Tensor result = empty({0}, kLong, Device(kCPU));

  auto dtype = dets.scalar_type();
  if (dtype == kFloat) {
    result = nms_kernel_impl<float>(dets, scores, iou_threshold);
  } else if (dtype == kDouble) {
    result = nms_kernel_impl<double>(dets, scores, iou_threshold);
  } else {
    VISION_CHECK(false, "nms only supports float and double types");
  }
  return result;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("nms", TORCH_BOX(&nms_kernel));
}

} // namespace ops
} // namespace vision
