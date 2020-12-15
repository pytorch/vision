#pragma once

#include <torch/nn.h>
#include "../macros.h"

namespace vision {
namespace models {
struct VISION_API MobileNetV2Impl : torch::nn::Module {
  int64_t last_channel;
  torch::nn::Sequential features, classifier;

  explicit MobileNetV2Impl(
      int64_t num_classes = 1000,
      double width_mult = 1.0,
      std::vector<std::vector<int64_t>> inverted_residual_settings = {},
      int64_t round_nearest = 8);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MobileNetV2);
} // namespace models
} // namespace vision
