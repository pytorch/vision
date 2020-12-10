#pragma once

#include <torch/types.h>

C10_EXPORT torch::Tensor encodePNG(
    const torch::Tensor& data,
    int64_t compression_level);
