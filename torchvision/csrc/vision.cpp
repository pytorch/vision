// Autoregister all the torchvision ops (for C++).
#include "ops_autoregister.h"

#include <Python.h>

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_C(void) {
  // Register torchvision ops for python.
  vision::RegisterOps();
  return NULL;
}
#else
PyMODINIT_FUNC PyInit__C(void) {
  // Register torchvision ops for python.
  vision::RegisterOps();
  return NULL;
}
#endif
#endif
