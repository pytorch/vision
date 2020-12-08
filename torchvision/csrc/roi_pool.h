#pragma once

#include <ATen/ATen.h>
#include "macros.h"

namespace vision {
namespace ops {

VISION_API std::tuple<at::Tensor, at::Tensor> roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

} // namespace ops
} // namespace vision
