#pragma once

#include <torch/torch.h>
#include "../macros.h"

namespace vision {
namespace models {
// Densenet-BC model class, based on
// "Densely Connected Convolutional Networks"
// <https://arxiv.org/pdf/1608.06993.pdf>

// Args:
//     num_classes (int) - number of classification classes
//     growth_rate (int) - how many filters to add each layer (`k` in paper)
//     block_config (list of 4 ints) - how many layers in each pooling block
//     num_init_features (int) - the number of filters to learn in the first
//         convolution layer
//     bn_size (int) - multiplicative factor for number of bottle neck layers
//         (i.e. bn_size * k features in the bottleneck layer)
//     drop_rate (float) - dropout rate after each dense layer
struct VISION_API DenseNetImpl : torch::nn::Module {
  torch::nn::Sequential features{nullptr};
  torch::nn::Linear classifier{nullptr};

  explicit DenseNetImpl(
      int64_t num_classes = 1000,
      int64_t growth_rate = 32,
      const std::vector<int64_t>& block_config = {6, 12, 24, 16},
      int64_t num_init_features = 64,
      int64_t bn_size = 4,
      double drop_rate = 0);

  torch::Tensor forward(torch::Tensor x);
};

struct VISION_API DenseNet121Impl : DenseNetImpl {
  explicit DenseNet121Impl(
      int64_t num_classes = 1000,
      int64_t growth_rate = 32,
      const std::vector<int64_t>& block_config = {6, 12, 24, 16},
      int64_t num_init_features = 64,
      int64_t bn_size = 4,
      double drop_rate = 0);
};

struct VISION_API DenseNet169Impl : DenseNetImpl {
  explicit DenseNet169Impl(
      int64_t num_classes = 1000,
      int64_t growth_rate = 32,
      const std::vector<int64_t>& block_config = {6, 12, 32, 32},
      int64_t num_init_features = 64,
      int64_t bn_size = 4,
      double drop_rate = 0);
};

struct VISION_API DenseNet201Impl : DenseNetImpl {
  explicit DenseNet201Impl(
      int64_t num_classes = 1000,
      int64_t growth_rate = 32,
      const std::vector<int64_t>& block_config = {6, 12, 48, 32},
      int64_t num_init_features = 64,
      int64_t bn_size = 4,
      double drop_rate = 0);
};

struct VISION_API DenseNet161Impl : DenseNetImpl {
  explicit DenseNet161Impl(
      int64_t num_classes = 1000,
      int64_t growth_rate = 48,
      const std::vector<int64_t>& block_config = {6, 12, 36, 24},
      int64_t num_init_features = 96,
      int64_t bn_size = 4,
      double drop_rate = 0);
};

TORCH_MODULE(DenseNet);
TORCH_MODULE(DenseNet121);
TORCH_MODULE(DenseNet169);
TORCH_MODULE(DenseNet201);
TORCH_MODULE(DenseNet161);

} // namespace models
} // namespace vision
