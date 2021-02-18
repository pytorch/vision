#pragma once

#include <torch/types.h>
#include "../image_read_mode.h"

namespace vision {
namespace image {

C10_EXPORT torch::Tensor decode_png(
    const torch::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED);

} // namespace image
} // namespace vision
