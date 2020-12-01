#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

VISION_API at::Tensor nms_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);
