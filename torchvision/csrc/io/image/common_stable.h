#pragma once

#include <cstdint>

#include <torch/csrc/stable/tensor.h>

namespace vision {
namespace image {

/* Should be kept in-sync with Python ImageReadMode enum */
using ImageReadMode = int64_t;
const ImageReadMode IMAGE_READ_MODE_UNCHANGED = 0;
const ImageReadMode IMAGE_READ_MODE_GRAY = 1;
const ImageReadMode IMAGE_READ_MODE_GRAY_ALPHA = 2;
const ImageReadMode IMAGE_READ_MODE_RGB = 3;
const ImageReadMode IMAGE_READ_MODE_RGB_ALPHA = 4;

// Stable-ABI missing permute op; shim mirrors torchcodec's stablePermute:
// https://github.com/meta-pytorch/torchcodec/blob/1dc85b7a7900d91fee207ccdc02f211a051688fe/src/torchcodec/_core/StableABICompat.h#L81-L90
// TODO(stable-abi): remove once permute lands in the stable ABI upstream.
inline torch::stable::Tensor stable_permute(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dims)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::permute", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// Stable-ABI missing flip op, shimmed with the same dispatcher idiom as
// stable_permute.
// TODO(stable-abi): remove once flip lands in the stable ABI upstream.
inline torch::stable::Tensor stable_flip(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dims)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::flip", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

void validate_encoded_data(const torch::stable::Tensor& encoded_data);

} // namespace image
} // namespace vision
