#include "image.h"

#include <ATen/core/op_registration/op_registration.h>

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
