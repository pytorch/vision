#include "vision.h"

#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "ps_roi_pool.h"
#include "roi_align.h"
#include "roi_pool.h"

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
PyMODINIT_FUNC PyInit__C(void) {
  // No need to do anything.
  return NULL;
}
#endif

namespace vision {
int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}
} // namespace vision

using namespace vision::ops;

TORCH_LIBRARY(torchvision, m) {
  m.def(
      "ps_roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> (Tensor, Tensor)");
  m.def(
      "_ps_roi_pool_backward(Tensor grad, Tensor rois, Tensor channel_mapping, float spatial_scale, int pooled_height, int pooled_width, int batch_size, int channels, int height, int width) -> Tensor");
  m.def(
      "roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> Tensor");
  m.def(
      "_roi_align_backward(Tensor grad, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int batch_size, int channels, int height, int width, int sampling_ratio, bool aligned) -> Tensor");
  m.def(
      "roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> (Tensor, Tensor)");
  m.def(
      "_roi_pool_backward(Tensor grad, Tensor rois, Tensor argmax, float spatial_scale, int pooled_height, int pooled_width, int batch_size, int channels, int height, int width) -> Tensor");
  m.def("_cuda_version", &vision::cuda_version);
}

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("ps_roi_pool", ps_roi_pool_forward_cpu);
  m.impl("_ps_roi_pool_backward", ps_roi_pool_backward_cpu);
  m.impl("roi_align", roi_align_forward_cpu);
  m.impl("_roi_align_backward", roi_align_backward_cpu);
  m.impl("roi_pool", roi_pool_forward_cpu);
  m.impl("_roi_pool_backward", roi_pool_backward_cpu);
}

// TODO: Place this in a hypothetical separate torchvision_cuda library
#if defined(WITH_CUDA) || defined(WITH_HIP)
TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("ps_roi_pool", ps_roi_pool_forward_cuda);
  m.impl("_ps_roi_pool_backward", ps_roi_pool_backward_cuda);
  m.impl("roi_align", roi_align_forward_cuda);
  m.impl("_roi_align_backward", roi_align_backward_cuda);
  m.impl("roi_pool", roi_pool_forward_cuda);
  m.impl("_roi_pool_backward", roi_pool_backward_cuda);
}
#endif

// Autocast only needs to wrap forward pass ops.
#if defined(WITH_CUDA) || defined(WITH_HIP)
TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl("ps_roi_pool", ps_roi_pool_autocast);
  m.impl("roi_align", roi_align_autocast);
  m.impl("roi_pool", roi_pool_autocast);
}
#endif

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl("ps_roi_pool", ps_roi_pool_autograd);
  m.impl("_ps_roi_pool_backward", ps_roi_pool_backward_autograd);
  m.impl("roi_align", roi_align_autograd);
  m.impl("_roi_align_backward", roi_align_backward_autograd);
  m.impl("roi_pool", roi_pool_autograd);
  m.impl("_roi_pool_backward", roi_pool_backward_autograd);
}
