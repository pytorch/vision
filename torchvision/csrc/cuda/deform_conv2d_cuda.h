#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

VISION_API at::Tensor deform_conv2d_forward_cuda(
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
    deform_conv2d_backward_cuda(
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
