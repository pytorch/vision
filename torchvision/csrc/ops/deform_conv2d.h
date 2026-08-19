#pragma once

#include <torch/csrc/stable/tensor.h>
#include "../macros.h"

#include <tuple>

namespace vision {
namespace ops {

VISION_API torch::stable::Tensor deform_conv2d(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& offset,
    const torch::stable::Tensor& mask,
    const torch::stable::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask);

namespace detail {

std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor>
_deform_conv2d_backward(
    const torch::stable::Tensor& grad,
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& offset,
    const torch::stable::Tensor& mask,
    const torch::stable::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask);

} // namespace detail

} // namespace ops
} // namespace vision
