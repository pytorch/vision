#pragma once
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif
#ifdef WITH_HIP
#include "hip/vision_cuda.h"
#endif

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));
  if (dets.is_cuda()) {
#if defined(WITH_CUDA)
    if (dets.numel() == 0) {
      at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return nms_cuda(dets, scores, iou_threshold);
#elif defined(WITH_HIP)
    if (dets.numel() == 0) {
      at::cuda::HIPGuard device_guard(dets.device());
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
