#pragma once

#include <torch/csrc/stable/tensor.h>
#include "../common_stable.h"

namespace vision {
namespace image {

torch::stable::Tensor decode_webp(
    const torch::stable::Tensor& encoded_data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED);

} // namespace image
} // namespace vision
