// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Extension-level glue for the _C_stable extension. Holds the pieces that
// belong to the extension as a whole rather than to any single operator.

#include "vision.h"

#include <torch/csrc/stable/library.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

// If we are in a Windows environment, we need to define
// initialization functions for the _C_stable extension.
#if !defined(MOBILE) && defined(_WIN32)
void* PyInit__C_stable(void) {
  return nullptr;
}
#endif // !defined(MOBILE) && defined(_WIN32)

namespace vision {
int64_t cuda_version() {
#ifdef WITH_CUDA
  // looks like 12060, 12080, etc.
  return CUDA_VERSION;
#else
  return -1;
#endif
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def("_cuda_version() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(torchvision, CompositeExplicitAutograd, m) {
  m.impl("_cuda_version", TORCH_BOX(&cuda_version));
}
} // namespace vision
