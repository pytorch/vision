#include <Python.h>
#include <torch/script.h>

#include "ROIAlign.h"
#include "ROIPool.h"
#include "nms.h"

using namespace at;

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

static auto registry =
    torch::RegisterOperators()
        .op("torchvision::nms", &nms)
        .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor",
            &ROIAlign_forward)
        .op("torchvision::roi_pool", &ROIPool_forward);
