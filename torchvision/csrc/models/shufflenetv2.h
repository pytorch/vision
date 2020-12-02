#pragma once

#include <torch/torch.h>
#include "../macros.h"

namespace vision {
namespace models {

struct VISION_API ShuffleNetV2Impl : torch::nn::Module {
  std::vector<int64_t> _stage_out_channels;
  torch::nn::Sequential conv1{nullptr}, stage2, stage3, stage4, conv5{nullptr};
  torch::nn::Linear fc{nullptr};

  ShuffleNetV2Impl(
      const std::vector<int64_t>& stage_repeats,
      const std::vector<int64_t>& stage_out_channels,
      int64_t num_classes = 1000);

  torch::Tensor forward(torch::Tensor x);
};

struct VISION_API ShuffleNetV2_x0_5Impl : ShuffleNetV2Impl {
  explicit ShuffleNetV2_x0_5Impl(int64_t num_classes = 1000);
};

struct VISION_API ShuffleNetV2_x1_0Impl : ShuffleNetV2Impl {
  explicit ShuffleNetV2_x1_0Impl(int64_t num_classes = 1000);
};

struct VISION_API ShuffleNetV2_x1_5Impl : ShuffleNetV2Impl {
  explicit ShuffleNetV2_x1_5Impl(int64_t num_classes = 1000);
};

struct VISION_API ShuffleNetV2_x2_0Impl : ShuffleNetV2Impl {
  explicit ShuffleNetV2_x2_0Impl(int64_t num_classes = 1000);
};

TORCH_MODULE(ShuffleNetV2);
TORCH_MODULE(ShuffleNetV2_x0_5);
TORCH_MODULE(ShuffleNetV2_x1_0);
TORCH_MODULE(ShuffleNetV2_x1_5);
TORCH_MODULE(ShuffleNetV2_x2_0);

} // namespace models
} // namespace vision
