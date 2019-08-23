#include "PSROIAlign.h"
#include "PSROIPool.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "nms.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // TODO: remove nms from here since it is now registered
  //       and used as a PyTorch custom op
  m.def("nms", &nms, "non-maximum suppression");
  m.def("ps_roi_align_forward", &PSROIAlign_forward, "PSROIAlign_forward");
  m.def("ps_roi_align_backward", &PSROIAlign_backward, "PSROIAlign_backward");
  m.def("ps_roi_pool_forward", &PSROIPool_forward, "PSROIPool_forward");
  m.def("ps_roi_pool_backward", &PSROIPool_backward, "PSROIPool_backward");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
#ifdef WITH_CUDA
  m.attr("CUDA_VERSION") = CUDA_VERSION;
#endif
}
