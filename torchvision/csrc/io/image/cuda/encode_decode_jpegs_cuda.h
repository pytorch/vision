#pragma once

#include <torch/types.h>
#include "../common.h"
#include "decode_jpegs_cuda.h"
#include "encode_jpegs_cuda.h"

namespace vision {
namespace image {

/*

Fast jpeg decoding with CUDA.
A100+ GPUs have dedicated hardware support for jpeg decoding.

Args:
    - encoded_images (const std::vector<torch::Tensor>&): a vector of tensors
    containing the jpeg bitstreams to be decoded. Each tensor must have dtype
    torch.uint8 and device cpu
    - mode (ImageReadMode): IMAGE_READ_MODE_UNCHANGED, IMAGE_READ_MODE_GRAY and
IMAGE_READ_MODE_RGB are supported
    - device (torch::Device): The desired CUDA device to run the decoding on and
which will contain the output tensors

Returns:
    - decoded_images (std::vector<torch::Tensor>): a vector of torch::Tensors of
dtype torch.uint8 on the specified <device> containing the decoded images

Notes:
    - If a single image fails, the whole batch fails.
    - This function is thread-safe
*/
C10_EXPORT std::vector<torch::Tensor> decode_jpegs_cuda(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode,
    torch::Device device);

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
