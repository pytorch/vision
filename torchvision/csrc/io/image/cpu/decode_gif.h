#pragma once

#include <torch/types.h>

namespace vision {
namespace image {

// encoded_data tensor must be 1D uint8 and contiguous
C10_EXPORT torch::Tensor decode_gif(const torch::Tensor& encoded_data);

} // namespace image
} // namespace vision
