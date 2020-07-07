#pragma once
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#include "autocast.h"
#endif
#ifdef WITH_HIP
#include "hip/vision_cuda.h"
#endif

// nms dispatch nexus
at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  // roi_align (and internal pytorch ops) don't perform arg checking in
  // the dispatch nexus.  However, for nms, the checks are wordy and common
  // across backends.  Checking in the nexus seems preferable to duplicating
  // checks in cpu, cuda, and hip kernels.
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
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::nms", "")
                       .typed<decltype(nms)>();
  return op.call(
      dets,
      scores,
      iou_threshold);

// I don't know what to do with this.  I'd happily put it in hip/vision_cuda.h:nms_cuda,
// but the hip directory does not exist.
// #if defined(WITH_HIP)
//   if (dets.is_cuda()) {
//     if (dets.numel() == 0) {
//       at::cuda::HIPGuard device_guard(dets.device());
//       return at::empty({0}, dets.options().dtype(at::kLong));
//     }
//     return nms_cuda(dets, scores, iou_threshold);
// #endif
}

#ifdef WITH_CUDA
at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return nms(
      autocast::_cast(at::kFloat, dets),
      autocast::_cast(at::kFloat, scores),
      iou_threshold);
}
#endif
