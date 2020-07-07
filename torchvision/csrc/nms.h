#pragma once
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "autocast.h"
#include "cuda/vision_cuda.h"
#endif
#ifdef WITH_HIP
#include "hip/vision_cuda.h"
#endif

// nms dispatch nexus
at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::nms", "")
                       .typed<decltype(nms)>();
  return op.call(dets, scores, iou_threshold);
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
