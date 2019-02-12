#ifndef VGG_H
#define VGG_H

#include <torch/torch.h>

namespace vision {
namespace models {
class VGGImpl : public torch::nn::Module {
  torch::nn::Sequential features{nullptr}, classifier{nullptr};

  void _initialize_weights();

 public:
  VGGImpl(
      torch::nn::Sequential features,
      int64_t num_classes = 1000,
      bool initialize_weights = true);

  torch::Tensor forward(torch::Tensor x);
};

// VGG 11-layer model (configuration "A")
class VGG11Impl : public VGGImpl {
 public:
  VGG11Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 13-layer model (configuration "B")
class VGG13Impl : public VGGImpl {
 public:
  VGG13Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 16-layer model (configuration "D")
class VGG16Impl : public VGGImpl {
 public:
  VGG16Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 19-layer model (configuration "E")
class VGG19Impl : public VGGImpl {
 public:
  VGG19Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 11-layer model (configuration "A") with batch normalization
class VGG11BNImpl : public VGGImpl {
 public:
  VGG11BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 13-layer model (configuration "B") with batch normalization
class VGG13BNImpl : public VGGImpl {
 public:
  VGG13BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 16-layer model (configuration "D") with batch normalization
class VGG16BNImpl : public VGGImpl {
 public:
  VGG16BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

// VGG 19-layer model (configuration 'E') with batch normalization
class VGG19BNImpl : public VGGImpl {
 public:
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
