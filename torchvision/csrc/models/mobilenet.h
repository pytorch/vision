#ifndef MOBILENET_H
#define MOBILENET_H

#include <torch/torch.h>

namespace vision {
namespace models {
struct MobileNetV2Impl : torch::nn::Module {
  int64_t last_channel;
  torch::nn::Sequential features, classifier;

  MobileNetV2Impl(int64_t num_classes = 1000, double width_mult = 1.0);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MobileNetV2);
} // namespace models
} // namespace vision

#endif // MOBILENET_H
