#pragma once

#include <torch/types.h>
#include "../image_read_mode.h"
#include "decode_jpegs_cuda.h"
#include "encode_jpegs_cuda.h"

namespace vision {
namespace image {

C10_EXPORT std::vector<torch::Tensor> decode_jpegs_cuda(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode,
    torch::Device device);

C10_EXPORT std::vector<torch::Tensor> encode_jpegs_cuda(
    const std::vector<torch::Tensor>& decoded_images,
    const int64_t quality);

} // namespace image
} // namespace vision
