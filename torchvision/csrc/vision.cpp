#include "vision.h"

#ifndef MOBILE
#include <Python.h>
#endif
#include <torch/library.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

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

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def("_cuda_version", &cuda_version);
}
} // namespace vision
