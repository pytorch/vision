#pragma once

#include <torch/torch.h>
#include "image_read_mode.h"

C10_EXPORT torch::Tensor decodeJPEG(
    const torch::Tensor& data,
    ImageReadMode mode = IMAGE_READ_MODE_UNCHANGED);
