#pragma once

#include <torch/torch.h>

C10_EXPORT torch::Tensor decodeJPEG_cuda(const torch::Tensor& data);
