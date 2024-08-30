
#include "common.h"
#include <torch/torch.h>

namespace vision {
namespace image {

void validate_encoded_data(const torch::Tensor& encoded_data) {
  TORCH_CHECK(encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  TORCH_CHECK(
      encoded_data.dtype() == torch::kU8,
      "Input tensor must have uint8 data type, got ",
      encoded_data.dtype());
  TORCH_CHECK(
      encoded_data.dim() == 1 && encoded_data.numel() > 0,
      "Input tensor must be 1-dimensional and non-empty, got ",
      encoded_data.dim(),
      " dims  and ",
      encoded_data.numel(),
      " numels.");
}

} // namespace image
} // namespace vision