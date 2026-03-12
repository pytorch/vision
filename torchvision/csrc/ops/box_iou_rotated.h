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

VISION_API at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);

} // namespace ops
} // namespace vision
