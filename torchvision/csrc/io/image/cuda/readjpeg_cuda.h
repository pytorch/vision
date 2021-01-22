#pragma once

#include <torch/types.h>

C10_EXPORT torch::Tensor decodeJPEG_cuda(const torch::Tensor& data);
