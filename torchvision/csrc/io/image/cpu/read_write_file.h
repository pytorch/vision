#pragma once

#include <torch/csrc/stable/tensor.h>

#include <string>

namespace vision {
namespace image {

torch::stable::Tensor read_file(const std::string& filename);

torch::stable::Tensor write_file(
    const std::string& filename,
    torch::stable::Tensor& data);

} // namespace image
} // namespace vision
