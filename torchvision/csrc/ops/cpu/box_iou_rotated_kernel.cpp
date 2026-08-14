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

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/TensorAccessor.h>

#include <cstdint>

#include "../box_iou_rotated_utils.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

template <typename T>
void box_iou_rotated_cpu_kernel(
    const Tensor& boxes1,
    const Tensor& boxes2,
    Tensor& ious) {
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  // Use accessors for efficient element access
  auto boxes1_a = torch::headeronly::HeaderOnlyTensorAccessor<const T, 2>(
      boxes1.const_data_ptr<T>(),
      boxes1.sizes().data(),
      boxes1.strides().data());
  auto boxes2_a = torch::headeronly::HeaderOnlyTensorAccessor<const T, 2>(
      boxes2.const_data_ptr<T>(),
      boxes2.sizes().data(),
      boxes2.strides().data());
  auto ious_a = torch::headeronly::HeaderOnlyTensorAccessor<float, 1>(
      ious.mutable_data_ptr<float>(),
      ious.sizes().data(),
      ious.strides().data());

  for (int64_t i = 0; i < num_boxes1; i++) {
    for (int64_t j = 0; j < num_boxes2; j++) {
      ious_a[i * num_boxes2 + j] =
          single_box_iou_rotated<T>(&boxes1_a[i][0], &boxes2_a[j][0]);
    }
  }
}

Tensor box_iou_rotated_cpu(
    // input must be contiguous:
    const Tensor& boxes1,
    const Tensor& boxes2) {
  STD_TORCH_CHECK(boxes1.is_cpu(), "boxes1 must be a CPU tensor");
  STD_TORCH_CHECK(boxes2.is_cpu(), "boxes2 must be a CPU tensor");
  // Stable-ABI sizes() is not streamable so the checks split to guard size(1):
  // https://github.com/meta-pytorch/torchcodec/blob/1dc85b7a7900d91fee207ccdc02f211a051688fe/src/torchcodec/_core/Encoder.cpp#L140-L147
  // TODO(stable-abi): print sizes() in the error message once it can be
  // streamed.
  STD_TORCH_CHECK(
      boxes1.dim() == 2,
      "boxes1 should be a 2d tensor, got ",
      boxes1.dim(),
      "D");
  STD_TORCH_CHECK(
      boxes1.size(1) == 5,
      "boxes1 should have 5 elements in dimension 1, got ",
      boxes1.size(1));
  STD_TORCH_CHECK(
      boxes2.dim() == 2,
      "boxes2 should be a 2d tensor, got ",
      boxes2.dim(),
      "D");
  STD_TORCH_CHECK(
      boxes2.size(1) == 5,
      "boxes2 should have 5 elements in dimension 1, got ",
      boxes2.size(1));
  STD_TORCH_CHECK(
      boxes1.scalar_type() == boxes2.scalar_type(),
      "boxes1 and boxes2 must have the same dtype");

  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  if (num_boxes1 == 0 || num_boxes2 == 0) {
    return torch::stable::new_empty(
        boxes1, {num_boxes1, num_boxes2}, torch::headeronly::ScalarType::Float);
  }

  auto boxes1_contiguous = torch::stable::contiguous(boxes1);
  auto boxes2_contiguous = torch::stable::contiguous(boxes2);

  Tensor ious = torch::stable::new_empty(
      boxes1, {num_boxes1 * num_boxes2}, torch::headeronly::ScalarType::Float);

  THO_DISPATCH_V2(
      boxes1.scalar_type(),
      "box_iou_rotated_cpu",
      AT_WRAP([&]() {
        box_iou_rotated_cpu_kernel<scalar_t>(
            boxes1_contiguous, boxes2_contiguous, ious);
      }),
      AT_EXPAND(AT_FLOATING_TYPES));

  return torch::stable::reshape(ious, {num_boxes1, num_boxes2});
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("box_iou_rotated", TORCH_BOX(&box_iou_rotated_cpu));
}

} // namespace ops
} // namespace vision
