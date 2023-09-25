#pragma once

#include <torch/types.h>
#include "../image_read_mode.h"

namespace vision {
namespace image {

C10_EXPORT torch::Tensor decode_jpeg(
    const torch::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED);

C10_EXPORT int64_t _jpeg_version();
C10_EXPORT bool _is_compiled_against_turbo();

} // namespace image
} // namespace vision
