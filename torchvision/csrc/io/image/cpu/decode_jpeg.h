#pragma once

#include <torch/csrc/stable/tensor.h>
#include "../common_stable.h"

namespace vision {
namespace image {

torch::stable::Tensor decode_jpeg(
    const torch::stable::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED,
    bool apply_exif_orientation = false);

int64_t _jpeg_version();
bool _is_compiled_against_turbo();

} // namespace image
} // namespace vision
