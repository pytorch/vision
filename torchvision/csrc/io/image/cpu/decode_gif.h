#pragma once

#include <torch/types.h> // TODO: Is this where C10_EXPORT is defined?? remove/change?

namespace vision {
namespace image {

C10_EXPORT torch::Tensor decode_gif(const torch::Tensor& encoded_data);

} // namespace image
} // namespace vision
