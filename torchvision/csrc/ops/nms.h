#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold,
    const double bias = 0.0);

} // namespace ops
} // namespace vision
