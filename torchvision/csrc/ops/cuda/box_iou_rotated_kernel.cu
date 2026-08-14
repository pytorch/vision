// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// This file contains code adapted from Detectron2's box_iou_rotated
// implementation, which is licensed under the Apache License, Version 2.0.
// Original source: https://github.com/facebookresearch/detectron2
// License: https://github.com/facebookresearch/detectron2/blob/main/LICENSE

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>

#include <utility>
#include <vector>

#include "../box_iou_rotated_utils.h"
#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

// 2D block with 32 * 16 = 512 threads per block
const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 16;

template <typename T>
__global__ void box_iou_rotated_cuda_kernel(
    const int n_boxes1,
    const int n_boxes2,
    const T* dev_boxes1,
    const T* dev_boxes2,
    float* dev_ious) {
  const int row_start = blockIdx.x * blockDim.x;
  const int col_start = blockIdx.y * blockDim.y;

  const int row_size = min(n_boxes1 - row_start, blockDim.x);
  const int col_size = min(n_boxes2 - col_start, blockDim.y);

  __shared__ T block_boxes1[BLOCK_DIM_X * 5];
  __shared__ T block_boxes2[BLOCK_DIM_Y * 5];

  // It's safe to copy using threadIdx.x since BLOCK_DIM_X >= BLOCK_DIM_Y
  if (threadIdx.x < row_size && threadIdx.y == 0) {
    block_boxes1[threadIdx.x * 5 + 0] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 0];
    block_boxes1[threadIdx.x * 5 + 1] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 1];
    block_boxes1[threadIdx.x * 5 + 2] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 2];
    block_boxes1[threadIdx.x * 5 + 3] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 3];
    block_boxes1[threadIdx.x * 5 + 4] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 4];
  }

  if (threadIdx.x < col_size && threadIdx.y == 0) {
    block_boxes2[threadIdx.x * 5 + 0] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 0];
    block_boxes2[threadIdx.x * 5 + 1] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 1];
    block_boxes2[threadIdx.x * 5 + 2] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 2];
    block_boxes2[threadIdx.x * 5 + 3] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 3];
    block_boxes2[threadIdx.x * 5 + 4] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size && threadIdx.y < col_size) {
    int offset = (row_start + threadIdx.x) * n_boxes2 + col_start + threadIdx.y;
    dev_ious[offset] = static_cast<float>(single_box_iou_rotated<T>(
        block_boxes1 + threadIdx.x * 5, block_boxes2 + threadIdx.y * 5));
  }
}

// THO_DISPATCH_V2 splits its body on commas outside parens. The commas in
// kernel<<<blocks, threads, 0, stream>>> would break it, so it goes through
// this wrapper.
template <typename scalar_t>
void launch_box_iou_rotated_cuda_kernel(
    dim3 blocks,
    dim3 threads,
    cudaStream_t stream,
    int n_boxes1,
    int n_boxes2,
    const scalar_t* boxes1,
    const scalar_t* boxes2,
    float* ious) {
  box_iou_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      n_boxes1, n_boxes2, boxes1, boxes2, ious);
}

Tensor box_iou_rotated_cuda(
    // input must be contiguous
    const Tensor& boxes1,
    const Tensor& boxes2) {
  STD_TORCH_CHECK(boxes1.is_cuda(), "boxes1 must be a CUDA tensor");
  STD_TORCH_CHECK(boxes2.is_cuda(), "boxes2 must be a CUDA tensor");
  STD_TORCH_CHECK(
      boxes1.scalar_type() == boxes2.scalar_type(),
      "boxes1 and boxes2 must have the same dtype");
  torch::stable::accelerator::DeviceGuard device_guard(
      boxes1.get_device_index());

  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  Tensor ious = torch::stable::new_empty(
      boxes1, {num_boxes1 * num_boxes2}, torch::headeronly::ScalarType::Float);

  bool transpose = false;
  if (num_boxes1 > 0 && num_boxes2 > 0) {
    if (num_boxes2 > 65535 * BLOCK_DIM_Y) {
      STD_TORCH_CHECK(
          num_boxes1 <= 65535 * BLOCK_DIM_Y,
          "Too many boxes for box_iou_rotated_cuda!");
      // x dim is allowed to be large, but y dim cannot,
      // so we transpose the two to avoid "invalid configuration argument"
      // error. We assume one of them is small. Otherwise the result is hard to
      // fit in memory anyway.
      std::swap(num_boxes1, num_boxes2);
      transpose = true;
    }

    const int blocks_x = ceil_div(static_cast<int>(num_boxes1), BLOCK_DIM_X);
    const int blocks_y = ceil_div(static_cast<int>(num_boxes2), BLOCK_DIM_Y);

    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
        boxes1.get_device_index(), &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    THO_DISPATCH_V2(
        boxes1.scalar_type(),
        "box_iou_rotated_cuda",
        AT_WRAP([&]() {
          const scalar_t* data1;
          const scalar_t* data2;
          if (transpose) {
            data1 = boxes2.const_data_ptr<scalar_t>();
            data2 = boxes1.const_data_ptr<scalar_t>();
          } else {
            data1 = boxes1.const_data_ptr<scalar_t>();
            data2 = boxes2.const_data_ptr<scalar_t>();
          }

          launch_box_iou_rotated_cuda_kernel<scalar_t>(
              blocks,
              threads,
              stream,
              num_boxes1,
              num_boxes2,
              data1,
              data2,
              ious.mutable_data_ptr<float>());
        }),
        AT_EXPAND(AT_FLOATING_TYPES));

    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // reshape from 1d array to 2d array
  auto shape = std::vector<int64_t>{num_boxes1, num_boxes2};
  if (transpose) {
    return torch::stable::transpose(torch::stable::view(ious, shape), 0, 1);
  } else {
    return torch::stable::view(ious, shape);
  }
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("box_iou_rotated", TORCH_BOX(&box_iou_rotated_cuda));
}

} // namespace ops
} // namespace vision
