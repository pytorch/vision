#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor deform_conv2d(
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
    int64_t groups,
    int64_t offset_groups,
    bool use_mask);

VISION_API at::Tensor deform_conv2d_symint(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    c10::SymInt stride_h,
    c10::SymInt stride_w,
    c10::SymInt pad_h,
    c10::SymInt pad_w,
    c10::SymInt dilation_h,
    c10::SymInt dilation_w,
    c10::SymInt groups,
    c10::SymInt offset_groups,
    bool use_mask);

namespace detail {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_deform_conv2d_backward(
    const at::Tensor& grad,
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
    int64_t groups,
    int64_t offset_groups,
    bool use_mask);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_deform_conv2d_backward_symint(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    c10::SymInt stride_h,
    c10::SymInt stride_w,
    c10::SymInt pad_h,
    c10::SymInt pad_w,
    c10::SymInt dilation_h,
    c10::SymInt dilation_w,
    c10::SymInt groups,
    c10::SymInt offset_groups,
    bool use_mask);

} // namespace detail

} // namespace ops
} // namespace vision
