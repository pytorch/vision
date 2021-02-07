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

namespace vision {
namespace image {

static auto registry = torch::RegisterOperators()
                           .op("image::decode_png", &decode_png)
                           .op("image::encode_png", &encode_png)
                           .op("image::decode_jpeg", &decode_jpeg)
                           .op("image::encode_jpeg", &encode_jpeg)
                           .op("image::read_file", &read_file)
                           .op("image::write_file", &write_file)
                           .op("image::decode_image", &decode_image)
                           .op("image::decode_jpeg_cuda", &decode_jpeg_cuda);

} // namespace image
} // namespace vision
