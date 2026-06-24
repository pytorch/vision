#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include "../../StableABICompat.h"

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

template <typename scalar_t>
Tensor qnms_kernel_impl(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {
  STD_TORCH_CHECK(!dets.is_cuda(), "dets must be a CPU tensor");
  STD_TORCH_CHECK(!scores.is_cuda(), "scores must be a CPU tensor");
  STD_TORCH_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return torch::stable::new_empty(
        dets, {0}, torch::headeronly::ScalarType::Long);
  }

  const auto ndets = dets.size(0);

  auto x1_t = torch::stable::contiguous(torch::stable::select(dets, 1, 0));
  auto y1_t = torch::stable::contiguous(torch::stable::select(dets, 1, 1));
  auto x2_t = torch::stable::contiguous(torch::stable::select(dets, 1, 2));
  auto y2_t = torch::stable::contiguous(torch::stable::select(dets, 1, 3));
  auto order_t = std::get<1>(stable_helpers::sort(
      scores, /*stable=*/true, /*dim=*/0, /*descending=*/true));
  Tensor suppressed_t = torch::stable::new_zeros(
      dets, {ndets}, torch::headeronly::ScalarType::Byte);
  Tensor keep_t = torch::stable::new_zeros(
      dets, {ndets}, torch::headeronly::ScalarType::Long);
  Tensor areas_t = torch::stable::new_zeros(
      dets, {ndets}, torch::headeronly::ScalarType::Float);

  auto suppressed = suppressed_t.mutable_data_ptr<uint8_t>();
  auto keep = keep_t.mutable_data_ptr<int64_t>();
  auto order = order_t.const_data_ptr<int64_t>();
  auto x1 = x1_t.const_data_ptr<scalar_t>();
  auto y1 = y1_t.const_data_ptr<scalar_t>();
  auto x2 = x2_t.const_data_ptr<scalar_t>();
  auto y2 = y2_t.const_data_ptr<scalar_t>();
  auto areas = areas_t.mutable_data_ptr<float>();

  for (int64_t i = 0; i < ndets; i++) {
    // Note 1: To get the exact area we'd need to multiply by scale**2, but this
    // would get canceled out in the computation of ovr below. So we leave that
    // out.
    // Note 2: degenerate boxes (x2 < x1 or y2 < y1) may underflow, although
    // integral promotion rules will likely prevent it (see
    // https://stackoverflow.com/questions/32959564/subtraction-of-two-unsigned-gives-signed
    // for more details).
    areas[i] = (static_cast<float>(x2[i]) - static_cast<float>(x1[i])) *
        (static_cast<float>(y2[i]) - static_cast<float>(y1[i]));
  }

  int64_t num_to_keep = 0;
  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }
    keep[num_to_keep++] = i;

    // We explicitly cast coordinates to float so that the code can be
    // vectorized.
    float ix1val = static_cast<float>(x1[i]);
    float iy1val = static_cast<float>(y1[i]);
    float ix2val = static_cast<float>(x2[i]);
    float iy2val = static_cast<float>(y2[i]);
    float iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
        continue;
      }
      float xx1 = std::max(ix1val, static_cast<float>(x1[j]));
      float yy1 = std::max(iy1val, static_cast<float>(y1[j]));
      float xx2 = std::min(ix2val, static_cast<float>(x2[j]));
      float yy2 = std::min(iy2val, static_cast<float>(y2[j]));

      auto w = std::max(0.f, xx2 - xx1); // * scale (gets canceled below)
      auto h = std::max(0.f, yy2 - yy1); // * scale (gets canceled below)
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  return torch::stable::narrow(
      keep_t, /*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

Tensor qnms_kernel(
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

  Tensor result =
      torch::stable::new_empty(dets, {0}, torch::headeronly::ScalarType::Long);
  THO_DISPATCH_V2(
      dets.scalar_type(),
      "qnms_kernel",
      AT_WRAP([&]() {
        result = qnms_kernel_impl<scalar_t>(dets, scores, iou_threshold);
      }),
      AT_EXPAND(AT_INTEGRAL_TYPES));
  return result;
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def("qnms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("qnms", TORCH_BOX(&qnms_kernel));
}

} // namespace ops
} // namespace vision
