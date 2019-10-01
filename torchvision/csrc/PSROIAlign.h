#pragma once

#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

#include <iostream>

std::tuple<at::Tensor, at::Tensor> PSROIAlign_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio) {
  if (input.type().is_cuda()) {
  std::cout << "Tensor is CUDA in PSROIAlign_forward" << std::endl;
#ifdef WITH_CUDA
    std::cout << "Using CUDA version of PSROIAlign_forward" << std::endl;
    return PSROIAlign_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  std::cout << "Using CPU version of PSROIAlign_forward" << std::endl;
  return PSROIAlign_forward_cpu(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio);
}

at::Tensor PSROIAlign_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& mapping_channel,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
  if (grad.type().is_cuda()) {
  std::cout << "Tensor is CUDA in PSROIAlign_backward" << std::endl;
#ifdef WITH_CUDA
    std::cout << "Using CUDA version of PSROIAlign_backward" << std::endl;
    return PSROIAlign_backward_cuda(
        grad,
        rois,
        mapping_channel,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        batch_size,
        channels,
        height,
        width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  std::cout << "Using CPU version of PSROIAlign_backward" << std::endl;
  return PSROIAlign_backward_cpu(
      grad,
      rois,
      mapping_channel,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      batch_size,
      channels,
      height,
      width);
}
