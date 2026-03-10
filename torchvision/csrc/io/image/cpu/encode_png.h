#pragma once

#include "../../../stable_abi_compat.h"

namespace vision {
namespace image {

torch::stable::Tensor encode_png(
    const torch::stable::Tensor& data,
    int64_t compression_level);

} // namespace image
} // namespace vision
