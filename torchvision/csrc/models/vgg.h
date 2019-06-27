#ifndef VGG_H
#define VGG_H

#include <torch/torch.h>
#include "general.h"

namespace vision {
namespace models {
struct VISION_API VGGImpl : torch::nn::Module {
  torch::nn::Sequential features{nullptr}, classifier{nullptr};

  void _initialize_weights();

  VGGImpl(
      torch::nn::Sequential features,
      int64_t num_classes = 1000,
      bool initialize_weights = true);

  torch::Tensor forward(torch::Tensor x);
};

// VGG 11-layer model (configuration "A")
struct VISION_API VGG11Impl : VGGImpl {
  VGG11Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 13-layer model (configuration "B")
struct VISION_API VGG13Impl : VGGImpl {
  VGG13Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 16-layer model (configuration "D")
struct VISION_API VGG16Impl : VGGImpl {
  VGG16Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 19-layer model (configuration "E")
struct VISION_API VGG19Impl : VGGImpl {
  VGG19Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 11-layer model (configuration "A") with batch normalization
struct VISION_API VGG11BNImpl : VGGImpl {
  VGG11BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 13-layer model (configuration "B") with batch normalization
struct VISION_API VGG13BNImpl : VGGImpl {
  VGG13BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 16-layer model (configuration "D") with batch normalization
struct VISION_API VGG16BNImpl : VGGImpl {
  VGG16BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 19-layer model (configuration 'E') with batch normalization
struct VISION_API VGG19BNImpl : VGGImpl {
  VGG19BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

TORCH_MODULE(VGG);

TORCH_MODULE(VGG11);
TORCH_MODULE(VGG13);
TORCH_MODULE(VGG16);
TORCH_MODULE(VGG19);

TORCH_MODULE(VGG11BN);
TORCH_MODULE(VGG13BN);
TORCH_MODULE(VGG16BN);
TORCH_MODULE(VGG19BN);

} // namespace models
} // namespace vision

#endif // VGG_H
