#include "vision.h"

#ifndef MOBILE
#ifdef USE_PYTHON
#include <Python.h>
#endif
#endif
#include <torch/library.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension.
// For PyMODINIT_FUNC to work, we need to include Python.h
#if !defined(MOBILE) && defined(_WIN32)
#ifdef USE_PYTHON
PyMODINIT_FUNC PyInit__C(void) {
  // No need to do anything.
  return NULL;
}
#endif // USE_PYTHON
#endif // !defined(MOBILE) && defined(_WIN32)

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
