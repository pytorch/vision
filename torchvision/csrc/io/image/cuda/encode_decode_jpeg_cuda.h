#pragma once

#include <torch/types.h>
#include "../image_read_mode.h"

#if NVJPEG_FOUND
#include <nvjpeg.h>

extern nvjpegHandle_t nvjpeg_handle;
extern std::once_flag nvjpeg_handle_creation_flag;
#endif

namespace vision {
namespace image {

C10_EXPORT torch::Tensor decode_jpeg_cuda(
    const torch::Tensor& data,
    ImageReadMode mode,
    torch::Device device);

C10_EXPORT std::vector<torch::Tensor> encode_jpeg_cuda(
    const std::vector<torch::Tensor>& images,
    const int64_t quality);

void nvjpeg_init();

} // namespace image
} // namespace vision
