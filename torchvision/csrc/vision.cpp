#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include <cuda.h>
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

int64_t _cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def("nms", nms);
    m.def("roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> Tensor",
        &roi_align);
    m.def("roi_pool", roi_pool);
    m.def("_new_empty_tensor_op", new_empty_tensor);
    m.def("ps_roi_align", ps_roi_align);
    m.def("ps_roi_pool", ps_roi_pool);
    m.def("deform_conv2d", deform_conv2d);
    m.def("_cuda_version", _cuda_version);
}
