// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// ===========================================================================
// PyTorch Stable ABI Compatibility Header for TorchVision
// ===========================================================================
//
// This header provides compatibility types and macros for using PyTorch's
// stable ABI API. It replaces the standard PyTorch C++ APIs (torch::, at::,
// c10::) with their stable ABI equivalents.
//
// Target PyTorch version: 2.11+
//
// Note: TORCH_TARGET_VERSION is set to 0x020b000000000000 (PyTorch 2.11) in
// CMakeLists.txt. This ensures we only use stable ABI features available in
// PyTorch 2.11+, providing forward compatibility when building against newer
// PyTorch versions.

// Include stable ABI headers
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

#include <array>
#include <vector>

// ===========================================================================
// Error Handling Macro
// ===========================================================================
// Replacement for TORCH_CHECK() that works with stable ABI.
// Uses STD_TORCH_CHECK from the stable ABI headers.
// Note: Unlike TORCH_CHECK, this always requires a message argument.

#define VISION_CHECK(cond, ...) STD_TORCH_CHECK(cond, __VA_ARGS__)

// ===========================================================================
// Type Aliases
// ===========================================================================
// Convenient aliases for stable ABI types in vision namespace

namespace vision {
namespace stable {

// Tensor types
using Tensor = torch::stable::Tensor;

// Device types
using Device = torch::stable::Device;
using DeviceType = torch::headeronly::DeviceType;
using DeviceIndex = torch::stable::accelerator::DeviceIndex;

// Scalar types (dtype)
using ScalarType = torch::headeronly::ScalarType;

// DeviceGuard for CUDA context management
using DeviceGuard = torch::stable::accelerator::DeviceGuard;

// Array reference type for sizes/strides
using IntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;

// Layout and MemoryFormat
using Layout = torch::headeronly::Layout;
using MemoryFormat = torch::headeronly::MemoryFormat;

// ===========================================================================
// Constants
// ===========================================================================

// Device type constants
constexpr auto kCPU = torch::headeronly::DeviceType::CPU;
constexpr auto kCUDA = torch::headeronly::DeviceType::CUDA;

// Scalar type constants (equivalents of at::kUInt8, at::kFloat32, etc.)
constexpr auto kByte = torch::headeronly::ScalarType::Byte;
constexpr auto kChar = torch::headeronly::ScalarType::Char;
constexpr auto kShort = torch::headeronly::ScalarType::Short;
constexpr auto kInt = torch::headeronly::ScalarType::Int;
constexpr auto kLong = torch::headeronly::ScalarType::Long;
constexpr auto kHalf = torch::headeronly::ScalarType::Half;
constexpr auto kFloat = torch::headeronly::ScalarType::Float;
constexpr auto kDouble = torch::headeronly::ScalarType::Double;
constexpr auto kBool = torch::headeronly::ScalarType::Bool;
constexpr auto kUInt16 = torch::headeronly::ScalarType::UInt16;

// Layout constants
constexpr auto kStrided = torch::headeronly::Layout::Strided;

// ===========================================================================
// Helper Functions - Tensor Creation
// ===========================================================================

// Stable version of at::empty()
inline Tensor empty(
    std::initializer_list<int64_t> sizes,
    ScalarType dtype,
    Device device) {
  std::vector<int64_t> sizesVec(sizes);
  return torch::stable::empty(
      IntArrayRef(sizesVec.data(), sizesVec.size()), dtype, kStrided, device);
}

// Overload taking a vector
inline Tensor empty(
    const std::vector<int64_t>& sizes,
    ScalarType dtype,
    Device device) {
  return torch::stable::empty(
      IntArrayRef(sizes.data(), sizes.size()), dtype, kStrided, device);
}

// Helper to create CPU tensors
inline Tensor emptyCPU(std::initializer_list<int64_t> sizes, ScalarType dtype) {
  return empty(sizes, dtype, Device(kCPU));
}

// Stable version of at::zeros() - creates via empty then zeros
inline Tensor zeros(
    std::initializer_list<int64_t> sizes,
    ScalarType dtype,
    Device device) {
  std::vector<int64_t> sizesVec(sizes);
  auto tensor = torch::stable::empty(
      IntArrayRef(sizesVec.data(), sizesVec.size()), dtype, kStrided, device);
  // Use dispatcher to call aten::zero_
  std::array<StableIValue, 1> stack{torch::stable::detail::from(tensor)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::zero_", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// ===========================================================================
// Helper Functions - Tensor Operations
// ===========================================================================

// Stable version of tensor.copy_(src)
inline void copy_(Tensor& dst, const Tensor& src) {
  torch::stable::copy_(dst, src);
}

// Stable version of tensor.to(device)
inline Tensor to(const Tensor& tensor, const Device& device) {
  return torch::stable::to(tensor, device);
}

// Stable version of tensor.narrow(dim, start, length)
inline Tensor narrow(
    Tensor tensor,
    int64_t dim,
    int64_t start,
    int64_t length) {
  return torch::stable::narrow(tensor, dim, start, length);
}

// Note: contiguous() is provided by torch::stable::contiguous() directly
// Do NOT define a vision::stable::contiguous wrapper as it conflicts with the
// default parameter in torch::stable::contiguous(tensor, memory_format =
// Contiguous)

// Stable version of tensor.select(dim, index) - from torch::stable::select
inline Tensor select(const Tensor& tensor, int64_t dim, int64_t index) {
  return torch::stable::select(tensor, dim, index);
}

// Helper for tensor.is_contiguous()
inline bool is_contiguous(const Tensor& tensor) {
  return tensor.is_contiguous();
}

// ===========================================================================
// Helper Functions - Dispatcher Wrappers
// ===========================================================================

// Stable version of tensor.sort() - returns (values, indices)
// Uses dispatcher to call aten::sort.stable
inline std::pair<Tensor, Tensor> sort(
    const Tensor& tensor,
    int64_t dim,
    bool descending) {
  std::array<StableIValue, 4> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(true), // stable sort
      torch::stable::detail::from(dim),
      torch::stable::detail::from(descending)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sort", "stable", stack.data(), TORCH_ABI_VERSION));
  return std::make_pair(
      torch::stable::detail::to<Tensor>(stack[0]),
      torch::stable::detail::to<Tensor>(stack[1]));
}

// Stable version of argsort - returns indices only
inline Tensor argsort(const Tensor& tensor, int64_t dim, bool descending) {
  auto [values, indices] = sort(tensor, dim, descending);
  return indices;
}

// Stable version of tensor.permute()
inline Tensor permute(
    const Tensor& tensor,
    std::initializer_list<int64_t> dims) {
  std::vector<int64_t> dimsVec(dims);
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(IntArrayRef(dimsVec.data(), dimsVec.size()))};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::permute", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of at::cat() - concatenates tensors along a dimension
inline Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(tensors), torch::stable::detail::from(dim)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::cat", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of at::clamp()
inline Tensor clamp(const Tensor& tensor, double min_val, double max_val) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(min_val),
      torch::stable::detail::from(max_val)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::clamp", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of at::floor()
inline Tensor floor(const Tensor& tensor) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(tensor)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::floor", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of at::ceil()
inline Tensor ceil(const Tensor& tensor) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(tensor)};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::ceil", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor.reshape()
inline Tensor reshape(const Tensor& tensor, const std::vector<int64_t>& shape) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(IntArrayRef(shape.data(), shape.size()))};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::reshape", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor.view()
inline Tensor view(const Tensor& tensor, const std::vector<int64_t>& shape) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(IntArrayRef(shape.data(), shape.size()))};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::view", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor.flatten(start_dim)
inline Tensor flatten(const Tensor& tensor, int64_t start_dim) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(start_dim),
      torch::stable::detail::from(static_cast<int64_t>(-1))};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::flatten", "using_ints", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Note: transpose() is provided by torch::stable::transpose()

// Stable version of tensor.zero_()
inline Tensor& zero_(Tensor& tensor) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(tensor)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::zero_", "", stack.data(), TORCH_ABI_VERSION));
  tensor = torch::stable::detail::to<Tensor>(stack[0]);
  return tensor;
}

// Stable version of tensor.addmm_(mat1, mat2)
inline Tensor& addmm_(Tensor& self, const Tensor& mat1, const Tensor& mat2) {
  std::array<StableIValue, 5> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(mat1),
      torch::stable::detail::from(mat2),
      torch::stable::detail::from(1.0), // beta
      torch::stable::detail::from(1.0)}; // alpha
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::addmm_", "", stack.data(), TORCH_ABI_VERSION));
  self = torch::stable::detail::to<Tensor>(stack[0]);
  return self;
}

// Stable version of at::zeros with vector sizes
inline Tensor zeros(
    const std::vector<int64_t>& sizes,
    ScalarType dtype,
    Device device) {
  auto tensor = torch::stable::empty(
      IntArrayRef(sizes.data(), sizes.size()), dtype, kStrided, device);
  std::array<StableIValue, 1> stack{torch::stable::detail::from(tensor)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::zero_", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of at::zeros_like()
inline Tensor zeros_like(const Tensor& tensor) {
  // Use dispatcher to call aten::zeros_like
  std::array<StableIValue, 1> stack{torch::stable::detail::from(tensor)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::zeros_like", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of at::ones_like() * val
inline Tensor ones_like_times(const Tensor& tensor, const Tensor& val) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(tensor), torch::stable::detail::from(val)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::mul", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor.sum(dims)
inline Tensor sum(const Tensor& tensor, const std::vector<int64_t>& dims) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(IntArrayRef(dims.data(), dims.size())),
      torch::stable::detail::from(false)}; // keepdim
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sum", "dim_IntList", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor + other (broadcasting add)
inline Tensor add(const Tensor& self, const Tensor& other) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(other),
      torch::stable::detail::from(1.0)}; // alpha
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::add", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor.index_select(dim, index)
inline Tensor index_select(
    const Tensor& tensor,
    int64_t dim,
    const Tensor& index) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(index)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::index_select", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor.masked_select(mask)
inline Tensor masked_select(const Tensor& tensor, const Tensor& mask) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(tensor), torch::stable::detail::from(mask)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::masked_select", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor.flip(dims)
inline Tensor flip(const Tensor& tensor, std::initializer_list<int64_t> dims) {
  std::vector<int64_t> dimsVec(dims);
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(tensor),
      torch::stable::detail::from(IntArrayRef(dimsVec.data(), dimsVec.size()))};
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::flip", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of torch.from_file() - read file into tensor
inline Tensor from_file(
    const std::string& filename,
    bool shared,
    int64_t size,
    ScalarType dtype) {
  std::array<StableIValue, 4> stack{
      torch::stable::detail::from(filename),
      torch::stable::detail::from(shared),
      torch::stable::detail::from(size),
      torch::stable::detail::from(dtype)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::from_file", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Note: squeeze() is provided by torch::stable::squeeze()

// Stable version of tensor.unsqueeze(dim)
inline Tensor unsqueeze(const Tensor& tensor, int64_t dim) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(tensor), torch::stable::detail::from(dim)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::unsqueeze", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Stable version of tensor.clone()
inline Tensor clone(const Tensor& tensor) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(tensor)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::clone", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Create empty tensor with ChannelsLast memory format
inline Tensor emptyCPUChannelsLast(
    const std::vector<int64_t>& sizes,
    ScalarType dtype) {
  return torch::stable::empty(
      IntArrayRef(sizes.data(), sizes.size()),
      dtype,
      kStrided,
      Device(kCPU),
      std::nullopt, // pin_memory
      MemoryFormat::ChannelsLast);
}

// ===========================================================================
// Helper Functions - Utility
// ===========================================================================

// Helper to get a human-readable name for a scalar type
inline const char* scalarTypeName(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Byte:
      return "uint8";
    case ScalarType::Char:
      return "int8";
    case ScalarType::Short:
      return "int16";
    case ScalarType::Int:
      return "int32";
    case ScalarType::Long:
      return "int64";
    case ScalarType::Half:
      return "float16";
    case ScalarType::Float:
      return "float32";
    case ScalarType::Double:
      return "float64";
    case ScalarType::Bool:
      return "bool";
    default:
      return "unknown";
  }
}

// Helper to get a human-readable name for a device type
inline const char* deviceTypeName(DeviceType dtype) {
  switch (dtype) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    default:
      return "unknown";
  }
}

// Helper to convert IntArrayRef to a string for error messages
inline std::string intArrayRefToString(const IntArrayRef& arr) {
  std::string result = "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += std::to_string(arr[i]);
  }
  result += "]";
  return result;
}

} // namespace stable

// ===========================================================================
// CUDA Accumulator Type Traits
// ===========================================================================
// Replacement for at::acc_type<T, is_cuda> that works without ATen headers
// Used in CUDA device code for accumulating sums with higher precision

#ifdef __CUDACC__
namespace cuda {

// Primary template - default accumulator type is the same as input
template <typename T, bool is_cuda>
struct acc_type {
  using type = T;
};

// Specializations for CUDA: Half uses float for accumulation
template <>
struct acc_type<__half, true> {
  using type = float;
};

// Float and double use themselves
template <>
struct acc_type<float, true> {
  using type = float;
};

template <>
struct acc_type<double, true> {
  using type = double;
};

// Helper alias for convenience
template <typename T, bool is_cuda>
using acc_type_t = typename acc_type<T, is_cuda>::type;

} // namespace cuda
#endif // __CUDACC__

} // namespace vision
