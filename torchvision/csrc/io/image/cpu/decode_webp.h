#pragma once

#include <torch/types.h>

namespace vision {
namespace image {

C10_EXPORT torch::Tensor decode_webp(const torch::Tensor& data);

} // namespace image
} // namespace vision
