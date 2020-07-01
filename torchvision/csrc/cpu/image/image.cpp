
#include "image.h"
#include <ATen/ATen.h>
#include <Python.h>

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_image(void) {
  // No need to do anything.
  return NULL;
}
#else
PyMODINIT_FUNC PyInit_image(void) {
  // No need to do anything.
  return NULL;
}
#endif
#endif

static auto registry =
    torch::RegisterOperators().op("image::decode_png", &decodePNG);
