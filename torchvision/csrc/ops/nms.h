#pragma once

#include <torch/csrc/stable/tensor.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API torch::stable::Tensor nms(
    const torch::stable::Tensor& dets,
    const torch::stable::Tensor& scores,
    double iou_threshold);

} // namespace ops
} // namespace vision
