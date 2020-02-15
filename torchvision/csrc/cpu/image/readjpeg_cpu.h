#pragma once

#include <torch/torch.h>

torch::Tensor decodeJPEG(const torch::Tensor& data);
