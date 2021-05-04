#pragma once

#include <torch/types.h>
#include "../image_read_mode.h"

namespace vision {
namespace image {

C10_EXPORT torch::Tensor decode_jpeg_cuda(
    const torch::Tensor& data,
    ImageReadMode mode,
    torch::Device device);

C10_EXPORT torch::Tensor decode_jpeg_batch_cuda(
    const torch::Tensor& data,
    ImageReadMode mode,
    torch::Device device,
    int64_t batch_size,
    int64_t height,
    int64_t width
    );

} // namespace image
} // namespace vision
