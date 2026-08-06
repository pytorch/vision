#pragma once

#include <torch/csrc/stable/tensor.h>
#include "../macros.h"

#include <tuple>

namespace vision {
namespace ops {

VISION_API std::tuple<torch::stable::Tensor, torch::stable::Tensor> roi_pool(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

namespace detail {

torch::stable::Tensor _roi_pool_backward(
    const torch::stable::Tensor& grad,
    const torch::stable::Tensor& rois,
    const torch::stable::Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);

} // namespace detail

} // namespace ops
} // namespace vision
