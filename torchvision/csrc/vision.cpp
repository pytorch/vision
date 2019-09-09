#include "ROIAlign.h"
#include "ROIPool.h"
#include "nms.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
#ifdef WITH_CUDA
  m.attr("CUDA_VERSION") = CUDA_VERSION;
#endif
}
