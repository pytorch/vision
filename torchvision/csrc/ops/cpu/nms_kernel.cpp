// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// The rotated box IoU computation is adapted from Detectron2's implementation,
// which is licensed under the Apache License, Version 2.0.
// Original source: https://github.com/facebookresearch/detectron2
// License: https://github.com/facebookresearch/detectron2/blob/main/LICENSE

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../box_iou_rotated_utils.h"

namespace vision {
namespace ops {

namespace {

template <typename scalar_t, typename IoUFunc>
at::Tensor nms_kernel_impl(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold,
    IoUFunc iou_func) {
  TORCH_CHECK(dets.is_cpu(), "dets must be a CPU tensor");
  TORCH_CHECK(scores.is_cpu(), "scores must be a CPU tensor");
  TORCH_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }
    keep[num_to_keep++] = i;

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
        continue;
      }

      auto ovr = iou_func(dets, i, j);
      if (ovr > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

template <typename scalar_t>
struct AABBIoU {
  scalar_t operator()(const at::Tensor& dets, int64_t i, int64_t j) const {
    auto x1 = dets.data_ptr<scalar_t>();
    auto ncols = dets.size(1);

    auto ix1 = x1[i * ncols + 0];
    auto iy1 = x1[i * ncols + 1];
    auto ix2 = x1[i * ncols + 2];
    auto iy2 = x1[i * ncols + 3];
    auto iarea = (ix2 - ix1) * (iy2 - iy1);

    auto jx1 = x1[j * ncols + 0];
    auto jy1 = x1[j * ncols + 1];
    auto jx2 = x1[j * ncols + 2];
    auto jy2 = x1[j * ncols + 3];
    auto jarea = (jx2 - jx1) * (jy2 - jy1);

    auto xx1 = std::max(ix1, jx1);
    auto yy1 = std::max(iy1, jy1);
    auto xx2 = std::min(ix2, jx2);
    auto yy2 = std::min(iy2, jy2);

    auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
    auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
    auto inter = w * h;
    return inter / (iarea + jarea - inter);
  }
};

template <typename scalar_t>
struct RotatedIoU {
  scalar_t operator()(const at::Tensor& dets, int64_t i, int64_t j) const {
    return single_box_iou_rotated<scalar_t>(
        dets[i].data_ptr<scalar_t>(), dets[j].data_ptr<scalar_t>());
  }
};

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_kernel", [&] {
    result = nms_kernel_impl<scalar_t>(
        dets, scores, iou_threshold, AABBIoU<scalar_t>{});
  });
  return result;
}

at::Tensor nms_rotated_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 5,
      "boxes should have 5 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_rotated_kernel", [&] {
    result = nms_kernel_impl<scalar_t>(
        dets, scores, iou_threshold, RotatedIoU<scalar_t>{});
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::nms_rotated"),
      TORCH_FN(nms_rotated_kernel));
}

} // namespace ops
} // namespace vision
