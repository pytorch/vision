#pragma once
#include <torch/extension.h>
#include "../macros.h"

// TODO: Delete this file once all the methods are gone

VISION_API std::tuple<at::Tensor, at::Tensor> roi_pool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

VISION_API at::Tensor roi_pool_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);
