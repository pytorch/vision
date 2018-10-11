#include "ROIPool.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
    m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
}