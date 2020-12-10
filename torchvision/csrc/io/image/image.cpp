#include "image.h"

#include <Python.h>

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
PyMODINIT_FUNC PyInit_image(void) {
  // No need to do anything.
  return NULL;
}
#endif

static auto registry = torch::RegisterOperators()
                           .op("image::decode_png", &decodePNG)
                           .op("image::encode_png", &encodePNG)
                           .op("image::decode_jpeg", &decodeJPEG)
                           .op("image::encode_jpeg", &encodeJPEG)
                           .op("image::read_file", &read_file)
                           .op("image::write_file", &write_file)
                           .op("image::decode_image", &decode_image);
