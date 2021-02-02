#pragma once

#include <torch/types.h>

namespace vision {
namespace image {

C10_EXPORT torch::Tensor read_file(const std::string& filename);

C10_EXPORT void write_file(const std::string& filename, torch::Tensor& data);

} // namespace image
} // namespace vision
