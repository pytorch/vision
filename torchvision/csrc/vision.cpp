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
  m.def(
      "deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups) -> Tensor");
  m.def(
      "_deform_conv2d_backward(Tensor grad, Tensor input, Tensor weight, Tensor offset, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor");
  m.def(
      "ps_roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> (Tensor, Tensor)");
  m.def(
      "_ps_roi_align_backward(Tensor grad, Tensor rois, Tensor channel_mapping, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, int batch_size, int channels, int height, int width) -> Tensor");
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
  m.def("_new_empty_tensor_op", &new_empty_tensor);
}

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("deform_conv2d", DeformConv2d_forward_cpu);
  m.impl("_deform_conv2d_backward", DeformConv2d_backward_cpu);
  m.impl("nms", nms_cpu);
  m.impl("ps_roi_align", PSROIAlign_forward_cpu);
  m.impl("_ps_roi_align_backward", PSROIAlign_backward_cpu);
  m.impl("ps_roi_pool", PSROIPool_forward_cpu);
  m.impl("_ps_roi_pool_backward", PSROIPool_backward_cpu);
  m.impl("roi_align", ROIAlign_forward_cpu);
  m.impl("_roi_align_backward", ROIAlign_backward_cpu);
  m.impl("roi_pool", ROIPool_forward_cpu);
  m.impl("_roi_pool_backward", ROIPool_backward_cpu);
}

// TODO: Place this in a hypothetical separate torchvision_cuda library
#if defined(WITH_CUDA) || defined(WITH_HIP)
TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("deform_conv2d", DeformConv2d_forward_cuda);
  m.impl("_deform_conv2d_backward", DeformConv2d_backward_cuda);
  m.impl("nms", nms_cuda);
  m.impl("ps_roi_align", PSROIAlign_forward_cuda);
  m.impl("_ps_roi_align_backward", PSROIAlign_backward_cuda);
  m.impl("ps_roi_pool", PSROIPool_forward_cuda);
  m.impl("_ps_roi_pool_backward", PSROIPool_backward_cuda);
  m.impl("roi_align", ROIAlign_forward_cuda);
  m.impl("_roi_align_backward", ROIAlign_backward_cuda);
  m.impl("roi_pool", ROIPool_forward_cuda);
  m.impl("_roi_pool_backward", ROIPool_backward_cuda);
}
#endif

// Autocast only needs to wrap forward pass ops.
#if defined(WITH_CUDA) || defined(WITH_HIP)
TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl("deform_conv2d", DeformConv2d_autocast);
  m.impl("nms", nms_autocast);
  m.impl("ps_roi_align", PSROIAlign_autocast);
  m.impl("ps_roi_pool", PSROIPool_autocast);
  m.impl("roi_align", ROIAlign_autocast);
  m.impl("roi_pool", ROIPool_autocast);
}
#endif

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl("deform_conv2d", DeformConv2d_autograd);
  m.impl("_deform_conv2d_backward", DeformConv2d_backward_autograd);
  m.impl("ps_roi_align", PSROIAlign_autograd);
  m.impl("_ps_roi_align_backward", PSROIAlign_backward_autograd);
  m.impl("ps_roi_pool", PSROIPool_autograd);
  m.impl("_ps_roi_pool_backward", PSROIPool_backward_autograd);
  m.impl("roi_align", ROIAlign_autograd);
  m.impl("_roi_align_backward", ROIAlign_backward_autograd);
  m.impl("roi_pool", ROIPool_autograd);
  m.impl("_roi_pool_backward", ROIPool_backward_autograd);
}
