#pragma once

#include <torch/torch.h>

C10_EXPORT torch::Tensor decode_image(
    const torch::Tensor& data,
    int64_t mode = 0);
