#pragma once

#include <torch/torch.h>
#include "../macros.h"

namespace vision {
namespace models {
// AlexNet model architecture from the
// "One weird trick..." <https://arxiv.org/abs/1404.5997> paper.
struct VISION_API AlexNetImpl : torch::nn::Module {
  torch::nn::Sequential features{nullptr}, classifier{nullptr};

  explicit AlexNetImpl(int64_t num_classes = 1000);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(AlexNet);

} // namespace models
} // namespace vision
