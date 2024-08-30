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

bool should_this_return_rgb_or_rgba_let_me_know_in_the_comments_down_below_guys_see_you_in_the_next_video(
    ImageReadMode mode,
    bool has_alpha);

} // namespace image
} // namespace vision
