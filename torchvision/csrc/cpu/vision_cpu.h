#pragma once
#include <torch/extension.h>
#include "../macros.h"

VISION_API at::Tensor deform_conv2d_forward_cpu(
    const at::Tensor& input_param,
    const at::Tensor& weight_param,
    const at::Tensor& offset_param,
    const at::Tensor& mask_param,
    const at::Tensor& bias_param,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask);

VISION_API std::
    tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    deform_conv2d_backward_cpu(
        const at::Tensor& grad_out_param,
        const at::Tensor& input_param,
        const at::Tensor& weight_param,
        const at::Tensor& offset_param,
        const at::Tensor& mask_param,
        const at::Tensor& bias_param,
        int64_t stride_h,
        int64_t stride_w,
        int64_t pad_h,
        int64_t pad_w,
        int64_t dil_h,
        int64_t dil_w,
        int64_t n_weight_grps,
        int64_t n_offset_grps,
        bool use_mask);

VISION_API at::Tensor nms_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

VISION_API std::tuple<at::Tensor, at::Tensor> PSROIAlign_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio);

VISION_API at::Tensor PSROIAlign_backward_cpu(
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

VISION_API std::tuple<at::Tensor, at::Tensor> PSROIPool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

VISION_API at::Tensor PSROIPool_backward_cpu(
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

VISION_API at::Tensor ROIAlign_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned);

VISION_API at::Tensor ROIAlign_backward_cpu(
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

VISION_API std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

VISION_API at::Tensor ROIPool_backward_cpu(
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
