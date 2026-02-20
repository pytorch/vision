#pragma once

#include "../../../StableABICompat.h"

namespace vision {
namespace image {

vision::stable::Tensor encode_png(
    const vision::stable::Tensor& data,
    int64_t compression_level);

} // namespace image
} // namespace vision
