#ifndef MNASNET_H
#define MNASNET_H

#include <torch/torch.h>
#include "general.h"

namespace vision {
namespace models {
struct VISION_API MNASNetImpl : torch::nn::Module {
  torch::nn::Sequential layers, classifier;

  void _initialize_weights();

  MNASNetImpl(double alpha, int64_t num_classes = 1000, double dropout = .2);

  torch::Tensor forward(torch::Tensor x);
};

struct MNASNet0_5Impl : MNASNetImpl {
  MNASNet0_5Impl(int64_t num_classes = 1000, double dropout = .2);
};

struct MNASNet0_75Impl : MNASNetImpl {
  MNASNet0_75Impl(int64_t num_classes = 1000, double dropout = .2);
};

struct MNASNet1_0Impl : MNASNetImpl {
  MNASNet1_0Impl(int64_t num_classes = 1000, double dropout = .2);
};

struct MNASNet1_3Impl : MNASNetImpl {
  MNASNet1_3Impl(int64_t num_classes = 1000, double dropout = .2);
};

TORCH_MODULE(MNASNet);
TORCH_MODULE(MNASNet0_5);
TORCH_MODULE(MNASNet0_75);
TORCH_MODULE(MNASNet1_0);
TORCH_MODULE(MNASNet1_3);

} // namespace models
} // namespace vision

#endif // MNASNET_H
