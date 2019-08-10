#ifndef READPNG_H
#define READPNG_H

#include <torch/torch.h>

namespace torch {
namespace vision {
namespace image {
namespace impl {

bool is_png(const void* data);
torch::Tensor readpng(const void* data);

} // namespace impl
} // namespace image
} // namespace vision
} // namespace torch

#endif // READPNG_H
