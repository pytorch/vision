#pragma once
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    if (dets.numel() == 0) {
      at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return nms_cuda(dets, scores, iou_threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = nms_cpu(dets, scores, iou_threshold);
  return result;
}
