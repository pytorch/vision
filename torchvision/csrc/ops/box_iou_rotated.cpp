// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "box_iou_rotated.h"

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/headeronly/version.h>

#include <array>

namespace vision {
namespace ops {

using torch::stable::Tensor;

Tensor box_iou_rotated(const Tensor& boxes1, const Tensor& boxes2) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(boxes1), torch::stable::detail::from(boxes2)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::box_iou_rotated", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def("box_iou_rotated(Tensor boxes1, Tensor boxes2) -> Tensor");
}

} // namespace ops
} // namespace vision
