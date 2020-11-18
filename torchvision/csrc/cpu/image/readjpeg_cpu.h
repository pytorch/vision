#pragma once

#include <torch/torch.h>

C10_EXPORT torch::Tensor decodeJPEG(
    const torch::Tensor& data,
    int64_t channels = 0);
