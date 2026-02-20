#pragma once

#include "../../../StableABICompat.h"

namespace vision {
namespace image {

// encoded_data tensor must be 1D uint8 and contiguous
vision::stable::Tensor decode_gif(const vision::stable::Tensor& encoded_data);

} // namespace image
} // namespace vision
