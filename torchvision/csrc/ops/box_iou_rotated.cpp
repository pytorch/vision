// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "box_iou_rotated.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor box_iou_rotated(const at::Tensor& boxes1, const at::Tensor& boxes2) {
  C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.ops.box_iou_rotated.box_iou_rotated");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::box_iou_rotated", "")
                       .typed<decltype(box_iou_rotated)>();
  return op.call(boxes1, boxes2);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::box_iou_rotated(Tensor boxes1, Tensor boxes2) -> Tensor"));
}

} // namespace ops
} // namespace vision
