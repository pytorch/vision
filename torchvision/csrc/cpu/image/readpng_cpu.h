#pragma once

#include <torch/torch.h>

C10_EXPORT torch::Tensor decodePNG(const torch::Tensor& data, int64_t mode = 0);
