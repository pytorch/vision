#pragma once

#include "cpu/deform_conv2d_cpu.h"

#ifdef WITH_CUDA
#include "cuda/deform_conv2d_cuda.h"
#endif
#ifdef WITH_HIP
#include "hip/deform_conv2d_cuda.h"
#endif

// Autocast Registration
#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor deform_conv2d_autocast(
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
#endif

// Autograd Registration
at::Tensor deform_conv2d_autograd(
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
deform_conv2d_backward_autograd(
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
