#pragma once

#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor> PSROIPool_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width);

at::Tensor PSROIPool_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& mapping_channel,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width);
#endif

std::tuple<at::Tensor, at::Tensor> PSROIPool_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return PSROIPool_forward_cuda(
        input, rois, spatial_scale, pooled_height, pooled_width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor PSROIPool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& mapping_channel,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return PSROIPool_backward_cuda(
        grad,
        rois,
        mapping_channel,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
