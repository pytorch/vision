#pragma once

#include <torch/torch.h>

C10_API torch::Tensor encodeJPEG(const torch::Tensor& data, int64_t quality);
C10_API void writeJPEG(
    const torch::Tensor& data,
    std::string filename,
    int64_t quality);
