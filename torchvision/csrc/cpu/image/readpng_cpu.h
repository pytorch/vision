#pragma once

#include <torch/torch.h>
#include <string>

torch::Tensor decodePNG(const torch::Tensor& data);
