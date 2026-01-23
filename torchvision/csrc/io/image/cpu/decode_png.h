#pragma once

#include "../../../StableABICompat.h"
#include "../common.h"

namespace vision {
namespace image {

vision::stable::Tensor decode_png(
    const vision::stable::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED,
    bool apply_exif_orientation = false);

} // namespace image
} // namespace vision
