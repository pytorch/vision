#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor ROIAlign_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
#else
    std::runtime_error("Not compiled with GPU support");
#endif
  }
  return ROIAlign_forward_cpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

at::Tensor ROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& input,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int sampling_ratio) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_backward_cuda(grad, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
#else
    std::runtime_error("Not compiled with GPU support");
#endif
  }
  std::runtime_error("Not implemented on the CPU");
}

