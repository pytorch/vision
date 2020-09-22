#pragma once

#include <torch/torch.h>

C10_API torch::Tensor encodeJPEG(const torch::Tensor& data, int64_t quality);
C10_API void writeJPEG(
    const torch::Tensor& data,
    const char* filename,
    int64_t quality);
