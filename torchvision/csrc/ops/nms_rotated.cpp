// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "nms_rotated.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor nms_rotated(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.nms_rotated.nms_rotated");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::nms_rotated", "")
                       .typed<decltype(nms_rotated)>();
  return op.call(dets, scores, iou_threshold);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::nms_rotated(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"));
}

} // namespace ops
} // namespace vision
