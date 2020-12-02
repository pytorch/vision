#pragma once
#include <torch/extension.h>
#include "../macros.h"

// TODO: Delete this file once all the methods are gone

VISION_API std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width);

VISION_API at::Tensor ROIPool_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width);
