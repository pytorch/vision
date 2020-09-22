#pragma once

// Comment
#include <torch/torch.h>
#include <string>

C10_API torch::Tensor decodePNG(const torch::Tensor& data);
