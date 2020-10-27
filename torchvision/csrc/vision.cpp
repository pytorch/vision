#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "DeformConv.h"
#include "PSROIAlign.h"
#include "PSROIPool.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "empty_tensor_op.h"
#include "nms.h"

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_C(void) {
  // No need to do anything.
  // extension.py will run on load
  return NULL;
}
#else
PyMODINIT_FUNC PyInit__C(void) {
  // No need to do anything.
  // extension.py will run on load
  return NULL;
}
#endif
#endif

namespace vision {
int64_t cuda_version() noexcept {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}
} // namespace vision

TORCH_LIBRARY(torchvision, m) {
  m.def("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor");
  m.def(
      "roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> Tensor");
  m.def(
      "_roi_align_backward(Tensor grad, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int batch_size, int channels, int height, int width, int sampling_ratio, bool aligned) -> Tensor");
  m.def("roi_pool", &roi_pool);
  m.def("_new_empty_tensor_op", &new_empty_tensor);
  m.def("ps_roi_align", &ps_roi_align);
  m.def("ps_roi_pool", &ps_roi_pool);
  m.def(
      "deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups) -> Tensor");
  m.def(
      "_deform_conv2d_backward(Tensor grad, Tensor input, Tensor weight, Tensor offset, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("_cuda_version", &vision::cuda_version);
}

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("roi_align", ROIAlign_forward_cpu);
  m.impl("_roi_align_backward", ROIAlign_backward_cpu);
  m.impl("deform_conv2d", DeformConv2d_forward_cpu);
  m.impl("_deform_conv2d_backward", DeformConv2d_backward_cpu);
  m.impl("nms", nms_cpu);
}

// TODO: Place this in a hypothetical separate torchvision_cuda library
#if defined(WITH_CUDA) || defined(WITH_HIP)
TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("roi_align", ROIAlign_forward_cuda);
  m.impl("_roi_align_backward", ROIAlign_backward_cuda);
  m.impl("deform_conv2d", DeformConv2d_forward_cuda);
  m.impl("_deform_conv2d_backward", DeformConv2d_backward_cuda);
  m.impl("nms", nms_cuda);
}
#endif

// Autocast only needs to wrap forward pass ops.
#if defined(WITH_CUDA) || defined(WITH_HIP)
TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl("roi_align", ROIAlign_autocast);
  m.impl("deform_conv2d", DeformConv2d_autocast);
  m.impl("nms", nms_autocast);
}
#endif

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl("roi_align", ROIAlign_autograd);
  m.impl("_roi_align_backward", ROIAlign_backward_autograd);
  m.impl("deform_conv2d", DeformConv2d_autograd);
  m.impl("_deform_conv2d_backward", DeformConv2d_backward_autograd);
}
