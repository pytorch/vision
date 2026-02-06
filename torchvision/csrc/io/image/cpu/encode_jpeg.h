#pragma once

#include "../../../StableABICompat.h"

namespace vision {
namespace image {

vision::stable::Tensor encode_jpeg(
    const vision::stable::Tensor& data,
    int64_t quality);

} // namespace image
} // namespace vision
