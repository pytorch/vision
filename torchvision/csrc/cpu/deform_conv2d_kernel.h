#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor deform_conv2d_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask);

VISION_API std::
    tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    deform_conv2d_backward_cpu(
        const at::Tensor& grad_out,
        const at::Tensor& input,
        const at::Tensor& weight,
        const at::Tensor& offset,
        const at::Tensor& mask,
        const at::Tensor& bias,
        int64_t stride_h,
        int64_t stride_w,
        int64_t pad_h,
        int64_t pad_w,
        int64_t dilation_h,
        int64_t dilation_w,
        int64_t n_weight_grps,
        int64_t n_offset_grps,
        bool use_mask);

} // namespace ops
} // namespace vision
