#include "image.h"

#include <ATen/core/op_registration/op_registration.h>

namespace vision {
namespace image {

static auto registry =
    torch::RegisterOperators()
        .op("image::decode_gif", &decode_gif)
        .op("image::decode_jpeg(Tensor data, int mode, bool apply_exif_orientation=False) -> Tensor",
            &decode_jpeg)
        .op("image::decode_webp(Tensor encoded_data, int mode) -> Tensor",
            &decode_webp)
        .op("image::_jpeg_version", &_jpeg_version)
        .op("image::_is_compiled_against_turbo", &_is_compiled_against_turbo);

} // namespace image
} // namespace vision
