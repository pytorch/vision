#include "vision.h"

#include <torch/library.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
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
