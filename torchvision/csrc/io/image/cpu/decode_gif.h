#pragma once

#include <torch/csrc/stable/tensor.h>

namespace vision {
namespace image {

// encoded_data tensor must be 1D uint8 and contiguous
torch::stable::Tensor decode_gif(const torch::stable::Tensor& encoded_data);

} // namespace image
} // namespace vision
