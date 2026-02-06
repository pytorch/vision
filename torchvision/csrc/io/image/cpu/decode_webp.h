#pragma once

#include "../../../StableABICompat.h"
#include "../common.h"

namespace vision {
namespace image {

vision::stable::Tensor decode_webp(
    const vision::stable::Tensor& encoded_data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED);

} // namespace image
} // namespace vision
