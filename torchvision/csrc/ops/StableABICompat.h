// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/headeronly/version.h>

#include <array>
#include <optional>
#include <tuple>

// Stable-ABI compatibility shims for torchvision's C++ operators.
//
// Some ATen ops our kernels use have no torch::stable wrapper in
// <torch/csrc/stable/ops.h> yet. The officially recommended approach is to call
// them through torch_call_dispatcher -- an ABI-stable call into an op outside
// the stable surface (the same pattern used by
// https://github.com/meta-pytorch/torchcodec/blob/8bbce656797c4f2b00feb2784ffe76e408be1e4c/src/torchcodec/_core/StableABICompat.h).
//
// This file GROWS as operators migrate: add a helper the first time an op needs
// an ATen call with no stable wrapper. Each is a thin shim -- delete it once an
// upstream torch::stable wrapper lands.
// TODO(stable-abi): upstream torch::stable wrappers for sort / index_select /
// masked_select so these helpers can be removed.

namespace vision {
namespace ops {
namespace stable_helpers {

using torch::stable::Tensor;

// aten::sort.stable(Tensor self, bool? stable, int dim=-1, bool
// descending=False)
//     -> (Tensor values, Tensor indices)
inline std::tuple<Tensor, Tensor> sort(
    const Tensor& self,
    bool stable,
    int64_t dim,
    bool descending) {
  std::array<StableIValue, 4> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(std::optional<bool>(stable)),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(descending)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sort", "stable", stack.data(), TORCH_ABI_VERSION));
  return {
      torch::stable::detail::to<Tensor>(stack[0]),
      torch::stable::detail::to<Tensor>(stack[1])};
}

// aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
inline Tensor index_select(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(index)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::index_select", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// aten::masked_select(Tensor self, Tensor mask) -> Tensor
inline Tensor masked_select(const Tensor& self, const Tensor& mask) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(mask)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::masked_select", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

} // namespace stable_helpers
} // namespace ops
} // namespace vision
