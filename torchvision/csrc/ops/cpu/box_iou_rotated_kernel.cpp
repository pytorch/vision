// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// This file contains code adapted from Detectron2's box_iou_rotated
// implementation, which is licensed under the Apache License, Version 2.0.
// Original source: https://github.com/facebookresearch/detectron2
// License: https://github.com/facebookresearch/detectron2/blob/main/LICENSE

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../box_iou_rotated_utils.h"

namespace vision {
namespace ops {

namespace {

template <typename T>
void box_iou_rotated_cpu_kernel(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    at::Tensor& ious) {
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  // Use accessors for efficient element access
  auto boxes1_a = boxes1.accessor<T, 2>();
  auto boxes2_a = boxes2.accessor<T, 2>();
  auto ious_a = ious.accessor<float, 1>();

  for (int64_t i = 0; i < num_boxes1; i++) {
    for (int64_t j = 0; j < num_boxes2; j++) {
      ious_a[i * num_boxes2 + j] =
          single_box_iou_rotated<T>(&boxes1_a[i][0], &boxes2_a[j][0]);
    }
  }
}

at::Tensor box_iou_rotated_cpu(
    // input must be contiguous:
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  TORCH_CHECK(boxes1.is_cpu(), "boxes1 must be a CPU tensor");
  TORCH_CHECK(boxes2.is_cpu(), "boxes2 must be a CPU tensor");
  TORCH_CHECK(
      boxes1.dim() == 2 && boxes1.size(1) == 5,
      "boxes1 should have shape (N, 5), got ",
      boxes1.sizes());
  TORCH_CHECK(
      boxes2.dim() == 2 && boxes2.size(1) == 5,
      "boxes2 should have shape (M, 5), got ",
      boxes2.sizes());
  TORCH_CHECK(
      boxes1.scalar_type() == boxes2.scalar_type(),
      "boxes1 and boxes2 must have the same dtype");

  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  if (num_boxes1 == 0 || num_boxes2 == 0) {
    return at::empty(
        {num_boxes1, num_boxes2}, boxes1.options().dtype(at::kFloat));
  }

  auto boxes1_contiguous = boxes1.contiguous();
  auto boxes2_contiguous = boxes2.contiguous();

  at::Tensor ious =
      at::empty({num_boxes1 * num_boxes2}, boxes1.options().dtype(at::kFloat));

  AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "box_iou_rotated_cpu", [&] {
    box_iou_rotated_cpu_kernel<scalar_t>(
        boxes1_contiguous, boxes2_contiguous, ious);
  });

  return ious.reshape({num_boxes1, num_boxes2});
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::box_iou_rotated"),
      TORCH_FN(box_iou_rotated_cpu));
}

} // namespace ops
} // namespace vision
