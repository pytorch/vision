#pragma once

#include "../../../StableABICompat.h"
#include "../common.h"

namespace vision {
namespace image {

vision::stable::Tensor decode_jpeg(
    const vision::stable::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED,
    bool apply_exif_orientation = false);

int64_t _jpeg_version();
bool _is_compiled_against_turbo();

} // namespace image
} // namespace vision
