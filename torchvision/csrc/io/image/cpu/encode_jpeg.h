#pragma once

#include <torch/types.h>

namespace vision {
namespace image {

C10_EXPORT torch::Tensor encode_jpeg(
    const torch::Tensor& data,
    int64_t quality);

} // namespace image
} // namespace vision
