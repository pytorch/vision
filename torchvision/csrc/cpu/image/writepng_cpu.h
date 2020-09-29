#pragma once

#include <torch/torch.h>

C10_EXPORT torch::Tensor encodePNG(
    const torch::Tensor& data,
    int64_t compression_level);
C10_EXPORT void writePNG(
    const torch::Tensor& data,
    std::string filename,
    int64_t compression_level);
