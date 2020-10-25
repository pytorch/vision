#pragma once

#include <torch/torch.h>

C10_EXPORT torch::Tensor decodeJPEG(const torch::Tensor& data);
