#include "vision.h"

#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "deform_conv2d.h"
#include "new_empty_tensor_op.h"
#include "nms.h"
#include "ps_roi_align.h"
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
      "deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> Tensor");
  m.def(
      "_deform_conv2d_backward(Tensor grad, Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
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
  m.impl("deform_conv2d", deform_conv2d_forward_cpu);
  m.impl("_deform_conv2d_backward", deform_conv2d_backward_cpu);
  m.impl("nms", nms_cpu);
  m.impl("ps_roi_align", ps_roi_align_forward_cpu);
  m.impl("_ps_roi_align_backward", ps_roi_align_backward_cpu);
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
  m.impl("deform_conv2d", deform_conv2d_forward_cuda);
  m.impl("_deform_conv2d_backward", deform_conv2d_backward_cuda);
  m.impl("nms", nms_cuda);
  m.impl("ps_roi_align", ps_roi_align_forward_cuda);
  m.impl("_ps_roi_align_backward", ps_roi_align_backward_cuda);
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
  m.impl("deform_conv2d", deform_conv2d_autocast);
  m.impl("nms", nms_autocast);
  m.impl("ps_roi_align", ps_roi_align_autocast);
  m.impl("ps_roi_pool", ps_roi_pool_autocast);
  m.impl("roi_align", roi_align_autocast);
  m.impl("roi_pool", roi_pool_autocast);
}
#endif

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl("deform_conv2d", deform_conv2d_autograd);
  m.impl("_deform_conv2d_backward", deform_conv2d_backward_autograd);
  m.impl("ps_roi_align", ps_roi_align_autograd);
  m.impl("_ps_roi_align_backward", ps_roi_align_backward_autograd);
  m.impl("ps_roi_pool", ps_roi_pool_autograd);
  m.impl("_ps_roi_pool_backward", ps_roi_pool_backward_autograd);
  m.impl("roi_align", roi_align_autograd);
  m.impl("_roi_align_backward", roi_align_backward_autograd);
  m.impl("roi_pool", roi_pool_autograd);
  m.impl("_roi_pool_backward", roi_pool_backward_autograd);
}
