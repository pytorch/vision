#pragma once
#include <torch/extension.h>
#include "../macros.h"

// TODO: Delete this file once all the methods are gone

VISION_API at::Tensor nms_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

VISION_API std::tuple<at::Tensor, at::Tensor> PSROIAlign_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio);

VISION_API at::Tensor PSROIAlign_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);

VISION_API std::tuple<at::Tensor, at::Tensor> PSROIPool_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

VISION_API at::Tensor PSROIPool_backward_cuda(
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

VISION_API at::Tensor ROIAlign_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned);

VISION_API at::Tensor ROIAlign_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned);

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
