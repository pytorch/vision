#pragma once

#include <torch/types.h>
#include "../common.h"
#include "encode_jpegs_cuda.h"

namespace vision {
namespace image {

/*
Fast jpeg encoding with CUDA.

Args:
    - decoded_images (const std::vector<torch::Tensor>&): a vector of contiguous
CUDA tensors of dtype torch.uint8 to be encoded.
    - quality (int64_t): 0-100, 75 is the default

Returns:
    - encoded_images (std::vector<torch::Tensor>): a vector of CUDA
torch::Tensors of dtype torch.uint8 containing the encoded images

Notes:
    - If a single image fails, the whole batch fails.
    - This function is thread-safe
*/
C10_EXPORT std::vector<torch::Tensor> encode_jpegs_cuda(
    const std::vector<torch::Tensor>& decoded_images,
    const int64_t quality);

} // namespace image
} // namespace vision
