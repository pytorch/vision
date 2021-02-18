#pragma once

#include <torch/nn.h>

namespace vision {
namespace models {
namespace modelsimpl {

// TODO here torch::relu_ and torch::adaptive_avg_pool2d wrapped in
// torch::nn::Fuctional don't work. so keeping these for now

inline torch::Tensor& relu_(const torch::Tensor& x) {
  return x.relu_();
}

inline torch::Tensor& relu6_(const torch::Tensor& x) {
  return x.clamp_(0, 6);
}

inline torch::Tensor adaptive_avg_pool2d(
    const torch::Tensor& x,
    torch::ExpandingArray<2> output_size) {
  return torch::adaptive_avg_pool2d(x, output_size);
}

inline torch::Tensor max_pool2d(
    const torch::Tensor& x,
    torch::ExpandingArray<2> kernel_size,
    torch::ExpandingArray<2> stride) {
  return torch::max_pool2d(x, kernel_size, stride);
}

inline bool double_compare(double a, double b) {
  return double(std::abs(a - b)) < std::numeric_limits<double>::epsilon();
};

} // namespace modelsimpl
} // namespace models
} // namespace vision
