#pragma once

#include <torch/types.h>
#include "../image_read_mode.h"
#include "encode_jpegs_cuda.h"

namespace vision {
namespace image {

C10_EXPORT torch::Tensor decode_jpeg_cuda(
    const torch::Tensor& data,
    ImageReadMode mode,
    torch::Device device);

C10_EXPORT std::vector<torch::Tensor> encode_jpegs_cuda(
    const std::vector<torch::Tensor>& decoded_images,
    const int64_t quality);

} // namespace image
} // namespace vision
