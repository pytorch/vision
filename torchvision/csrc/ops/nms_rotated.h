// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor nms_rotated(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

} // namespace ops
} // namespace vision
