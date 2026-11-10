#pragma once

#include <torch/csrc/stable/tensor.h>
#include "../common_stable.h"

namespace vision {
namespace image {

torch::stable::Tensor decode_png(
    const torch::stable::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED,
    bool apply_exif_orientation = false);

} // namespace image
} // namespace vision
