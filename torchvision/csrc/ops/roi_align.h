#pragma once

#include <torch/csrc/stable/tensor.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API torch::stable::Tensor roi_align(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned);

namespace detail {

torch::stable::Tensor _roi_align_backward(
    const torch::stable::Tensor& grad,
    const torch::stable::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned);

} // namespace detail

} // namespace ops
} // namespace vision
