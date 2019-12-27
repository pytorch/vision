#pragma once

#include <torch/torch.h>
#include <string>

torch::Tensor decodeJPEG(const torch::Tensor& data);
