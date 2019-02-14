#ifndef MODELSIMPL_H
#define MODELSIMPL_H

#include <torch/torch.h>

namespace vision {
namespace models {
namespace modelsimpl {

// TODO here torch::relu_ and torch::adaptive_avg_pool2d wrapped in
// torch::nn::Fuctional don't work. so keeping these for now

inline torch::Tensor relu_(torch::Tensor x) {
  return torch::relu_(x);
}

class AdaptiveAvgPool2DImpl : public torch::nn::Module {
  torch::ExpandingArray<2> output_size;

 public:
  AdaptiveAvgPool2DImpl(torch::ExpandingArray<2> output_size);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(AdaptiveAvgPool2D);

} // namespace modelsimpl
} // namespace models
} // namespace vision

#endif // MODELSIMPL_H
