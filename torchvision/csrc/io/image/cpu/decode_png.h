#pragma once

#include "../../../stable_abi_compat.h"
#include "../common.h"

namespace vision {
namespace image {

torch::stable::Tensor decode_png(
    const torch::stable::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED,
    bool apply_exif_orientation = false);

} // namespace image
} // namespace vision
