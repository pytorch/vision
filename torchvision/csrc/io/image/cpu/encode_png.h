#pragma once

#include <torch/csrc/stable/tensor.h>

namespace vision {
namespace image {

torch::stable::Tensor encode_png(
    const torch::stable::Tensor& data,
    int64_t compression_level);

} // namespace image
} // namespace vision
