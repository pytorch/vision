#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API std::tuple<at::Tensor, at::Tensor> ps_roi_pool_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

VISION_API at::Tensor ps_roi_pool_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);

} // namespace ops
} // namespace vision
