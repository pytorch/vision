#pragma once

#include <torch/nn.h>
#include "../macros.h"

namespace vision {
namespace models {
struct VISION_API SqueezeNetImpl : torch::nn::Module {
  int64_t num_classes;
  torch::nn::Sequential features{nullptr}, classifier{nullptr};

  explicit SqueezeNetImpl(double version = 1.0, int64_t num_classes = 1000);

  torch::Tensor forward(torch::Tensor x);
};

// SqueezeNet model architecture from the "SqueezeNet: AlexNet-level
// accuracy with 50x fewer parameters and <0.5MB model size"
// <https://arxiv.org/abs/1602.07360> paper.
struct VISION_API SqueezeNet1_0Impl : SqueezeNetImpl {
  explicit SqueezeNet1_0Impl(int64_t num_classes = 1000);
};

// SqueezeNet 1.1 model from the official SqueezeNet repo
// <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>.
// SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
// than SqueezeNet 1.0, without sacrificing accuracy.
struct VISION_API SqueezeNet1_1Impl : SqueezeNetImpl {
  explicit SqueezeNet1_1Impl(int64_t num_classes = 1000);
};

TORCH_MODULE(SqueezeNet);
TORCH_MODULE(SqueezeNet1_0);
TORCH_MODULE(SqueezeNet1_1);

} // namespace models
} // namespace vision
