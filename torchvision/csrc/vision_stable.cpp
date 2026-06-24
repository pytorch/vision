// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Extension-level glue for the stable-ABI _C_stable extension -- the stable
// counterpart of vision.cpp. Holds pieces that belong to the extension rather
// than to any single operator, and grows as more ops migrate.

// If we are in a Windows environment, we need to define
// initialization functions for the _C_stable extension.
#if !defined(MOBILE) && defined(_WIN32)
void* PyInit__C_stable(void) {
  return nullptr;
}
#endif // !defined(MOBILE) && defined(_WIN32)
