#pragma once

#include <torch/types.h>

namespace vision {
namespace image {

C10_EXPORT torch::Tensor encodePNG(
    const torch::Tensor& data,
    int64_t compression_level);

} // namespace image
} // namespace vision
