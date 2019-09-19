#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#include "ROIAlign.h"
#include "ROIPool.h"
#include "nms.h"

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_custom_ops(void) {
  // No need to do anything.
  // _custom_ops.py will run on load
  return NULL;
}
#else
PyMODINIT_FUNC PyInit__custom_ops(void) {
  // No need to do anything.
  // _custom_ops.py will run on load
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

static auto registry =
    torch::RegisterOperators()
        .op("torchvision::nms", &nms)
        .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor",
            &roi_align)
        .op("torchvision::roi_pool", &roi_pool)
        .op("torchvision::_cuda_version", &_cuda_version);
