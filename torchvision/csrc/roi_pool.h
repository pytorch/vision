#pragma once

#include "cpu/roi_pool_kernel.h"

#ifdef WITH_CUDA
#include "cuda/roi_pool_kernel.h"
#endif
#ifdef WITH_HIP
#include "hip/roi_pool_kernel.h"
#endif

namespace vision {
namespace ops {

// C++ Forward
std::tuple<at::Tensor, at::Tensor> roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

// Autocast Forward
#if defined(WITH_CUDA) || defined(WITH_HIP)
std::tuple<at::Tensor, at::Tensor> roi_pool_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);
#endif

// C++ Backward
at::Tensor _roi_pool_backward(
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

// Autograd Forward and Backward
std::tuple<at::Tensor, at::Tensor> roi_pool_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

at::Tensor roi_pool_backward_autograd(
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

} // namespace ops
} // namespace vision
