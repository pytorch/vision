#pragma once

#include <torch/types.h>
#include "../common.h"

namespace vision {
namespace image {

C10_EXPORT torch::Tensor decode_image(
    const torch::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED,
    bool apply_exif_orientation = false);

} // namespace image
} // namespace vision
