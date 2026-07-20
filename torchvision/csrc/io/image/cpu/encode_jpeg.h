#pragma once

#include <torch/csrc/stable/tensor.h>

namespace vision {
namespace image {

torch::stable::Tensor encode_jpeg(
    const torch::stable::Tensor& data,
    int64_t quality);

} // namespace image
} // namespace vision
