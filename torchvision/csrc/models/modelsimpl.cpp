#include "modelsimpl.h"

namespace vision {
namespace models {
namespace modelsimpl {

AdaptiveAvgPool2DImpl::AdaptiveAvgPool2DImpl(
    torch::ExpandingArray<2> output_size)
    : output_size(output_size) {}

torch::Tensor AdaptiveAvgPool2DImpl::forward(torch::Tensor x) {
  return torch::adaptive_avg_pool2d(x, output_size);
}

} // namespace modelsimpl
} // namespace models
} // namespace vision
