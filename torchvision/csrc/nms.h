#pragma once

#include "cpu/nms_kernel.h"

#ifdef WITH_CUDA
#include "cuda/nms_kernel.h"
#endif
#ifdef WITH_HIP
#include "hip/nms_kernel.h"
#endif

namespace vision {
namespace ops {

// C++ Forward
at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

// Autocast Forward
#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);
#endif

} // namespace ops
} // namespace vision
