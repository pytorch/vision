#pragma once

#include <torch/torch.h>

C10_EXPORT torch::Tensor encodeJPEG(const torch::Tensor& data, int64_t quality);
