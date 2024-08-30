#pragma once

#include <stdint.h>
#include <torch/torch.h>

namespace vision {
namespace image {

/* Should be kept in-sync with Python ImageReadMode enum */
using ImageReadMode = int64_t;
const ImageReadMode IMAGE_READ_MODE_UNCHANGED = 0;
const ImageReadMode IMAGE_READ_MODE_GRAY = 1;
const ImageReadMode IMAGE_READ_MODE_GRAY_ALPHA = 2;
const ImageReadMode IMAGE_READ_MODE_RGB = 3;
const ImageReadMode IMAGE_READ_MODE_RGB_ALPHA = 4;

void validate_encoded_data(const torch::Tensor& encoded_data);

} // namespace image
} // namespace vision
