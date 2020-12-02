#pragma once

#include "cpu/ps_roi_align_kernel.h"

#ifdef WITH_CUDA
#include "cuda/ps_roi_align_kernel.h"
#endif
#ifdef WITH_HIP
#include "hip/ps_roi_align_kernel.h"
#endif

namespace vision {
namespace ops {

// C++ Forward
std::tuple<at::Tensor, at::Tensor> ps_roi_align(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio);

// Autocast Forward
#if defined(WITH_CUDA) || defined(WITH_HIP)
std::tuple<at::Tensor, at::Tensor> ps_roi_align_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio);
#endif

// C++ Backward
at::Tensor _ps_roi_align_backward(
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

// Autograd Forward and Backward
std::tuple<at::Tensor, at::Tensor> ps_roi_align_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio);

at::Tensor ps_roi_align_backward_autograd(
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

} // namespace ops
} // namespace vision
