#ifndef ALEXNET_H
#define ALEXNET_H

#include <torch/torch.h>

namespace vision {
namespace models {
// AlexNet model architecture from the
// "One weird trick..." <https://arxiv.org/abs/1404.5997> paper.
class AlexNetImpl : public torch::nn::Module {
  torch::nn::Sequential features{nullptr}, classifier{nullptr};

 public:
  AlexNetImpl(int64_t num_classes = 1000);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(AlexNet);

} // namespace models
} // namespace vision

#endif // ALEXNET_H
