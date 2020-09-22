#pragma once

#include <torch/torch.h>

C10_API torch::Tensor decodeJPEG(const torch::Tensor& data);
