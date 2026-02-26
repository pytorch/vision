#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Exception.h>

#include <array>
#include <string>
#include <vector>

// Conversion helpers between at::Tensor and torch::stable::Tensor.
// These are used at migration boundaries where some code is on the old API
// and some is on the stable ABI.
#include <ATen/Tensor.h>

namespace vision {

inline torch::stable::Tensor toStableTensor(at::Tensor t) {
  return torch::stable::Tensor(
      reinterpret_cast<AtenTensorHandle>(new at::Tensor(std::move(t))));
}

inline at::Tensor fromStableTensor(const torch::stable::Tensor& t) {
  return *reinterpret_cast<at::Tensor*>(t.get());
}

// Dispatcher-based helpers for ops not yet in the stable ABI.
inline torch::stable::Tensor stablePermute(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dims)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::permute", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor stableFlip(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dims)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::flip", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

// Accessor helpers for torch::stable::Tensor, modeled after torchcodec's
// StableABICompat.h. These construct a HeaderOnlyTensorAccessor from the
// stable tensor's raw pointer, sizes, and strides.
template <typename T, size_t N>
torch::headeronly::HeaderOnlyTensorAccessor<T, N> mutableAccessor(
    torch::stable::Tensor& tensor) {
  return torch::headeronly::HeaderOnlyTensorAccessor<T, N>(
      tensor.mutable_data_ptr<T>(),
      tensor.sizes().data(),
      tensor.strides().data());
}

template <typename T, size_t N>
torch::headeronly::HeaderOnlyTensorAccessor<const T, N> constAccessor(
    const torch::stable::Tensor& tensor) {
  return torch::headeronly::HeaderOnlyTensorAccessor<const T, N>(
      tensor.const_data_ptr<T>(),
      tensor.sizes().data(),
      tensor.strides().data());
}

// Stable ABI version of validate_encoded_data.
inline void validate_encoded_data_stable(
    const torch::stable::Tensor& encoded_data) {
  STD_TORCH_CHECK(
      encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  STD_TORCH_CHECK(
      encoded_data.scalar_type() == torch::headeronly::ScalarType::Byte,
      "Input tensor must have uint8 data type.");
  STD_TORCH_CHECK(
      encoded_data.dim() == 1 && encoded_data.numel() > 0,
      "Input tensor must be 1-dimensional and non-empty.");
}

} // namespace vision
