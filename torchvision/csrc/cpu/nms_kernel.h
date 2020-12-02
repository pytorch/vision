#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor nms_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

} // namespace ops
} // namespace vision
