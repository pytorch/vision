// @nolint (improperly imported third-party code)
/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/cuda/KernelUtils.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <tuple>

#include "../StableABICompat.h"
#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

const int kMaxParallelImgs = 32;

// TODO(stable-abi): use torch::stable::add once it is added to the stable ABI.
Tensor add_tensors(const Tensor& self, const Tensor& other) {
  return torch::stable::subtract(self, other, -1.0);
}

inline unsigned int GET_THREADS() {
#ifdef WITH_HIP
  return 256;
#endif
  return 512;
}

inline unsigned int GET_BLOCKS(const unsigned int THREADS, const int64_t N) {
  int device = 0;
  STD_CUDA_CHECK(cudaGetDevice(&device));
  int max_grid_dim_x = 0;
  STD_CUDA_CHECK(
      cudaDeviceGetAttribute(&max_grid_dim_x, cudaDevAttrMaxGridDimX, device));
  int64_t kMaxGridNum = max_grid_dim_x;
  return (unsigned int)std::min(kMaxGridNum, (N + THREADS - 1) / THREADS);
}

template <typename scalar_t, typename index_t>
__device__ scalar_t bilinear_interpolate(
    const scalar_t* in,
    index_t height,
    index_t width,
    scalar_t h,
    scalar_t w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  index_t h_low = floor(h);
  index_t w_low = floor(w);
  index_t h_high = h_low + 1;
  index_t w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t, typename index_t>
__global__ void deformable_im2col_kernel(
    index_t n,
    const scalar_t* input_ptr,
    const scalar_t* offset_ptr,
    const scalar_t* mask_ptr,
    index_t height,
    index_t width,
    index_t weight_h,
    index_t weight_w,
    index_t pad_h,
    index_t pad_w,
    index_t stride_h,
    index_t stride_w,
    index_t dilation_h,
    index_t dilation_w,
    index_t batch_sz,
    index_t n_in_channels,
    index_t n_offset_grps,
    index_t out_h,
    index_t out_w,
    bool use_mask,
    scalar_t* columns_ptr) {
  CUDA_1D_KERNEL_LOOP_T(index, n, index_t) {
    const index_t out_x = index % out_w;
    const index_t out_y = (index / out_w) % out_h;
    const index_t out_b = (index / (out_w * out_h)) % batch_sz;
    const index_t in_c = index / (out_w * out_h * batch_sz);
    const index_t out_c = in_c * weight_h * weight_w;

    index_t c_per_offset_grp = n_in_channels / n_offset_grps;
    const index_t grp_idx = in_c / c_per_offset_grp;

    columns_ptr +=
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    input_ptr +=
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
        out_h * out_w;

    if (use_mask) {
      mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
          out_h * out_w;
    }

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const index_t mask_idx = i * weight_w + j;
        const index_t offset_idx = 2 * mask_idx;

        scalar_t mask_value = 1;
        if (use_mask) {
          mask_value =
              mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
        }

        const scalar_t offset_h =
            offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t offset_w = offset_ptr
            [(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t y =
            (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
        const scalar_t x =
            (out_x * stride_w - pad_w) + j * dilation_w + offset_w;
        *columns_ptr =
            mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

// THO_DISPATCH_V2 splits its body on commas outside parens. The commas in
// kernel<<<grid, block, 0, stream>>> and in the <scalar_t, index_t> template
// argument lists would break it, so the launches (including the int/int64
// indexing choice) go through these single-template-parameter wrappers.
template <typename scalar_t>
void launch_deformable_im2col_kernel(
    bool use_64bits_indexing,
    unsigned int blocks,
    unsigned int threads,
    cudaStream_t stream,
    int64_t num_kernels,
    const scalar_t* input,
    const scalar_t* offset,
    const scalar_t* mask,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int parallel_imgs,
    int n_in_channels,
    int deformable_group,
    int out_h,
    int out_w,
    bool use_mask,
    scalar_t* columns) {
  if (use_64bits_indexing) {
    deformable_im2col_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
        num_kernels,
        input,
        offset,
        mask,
        height,
        width,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        parallel_imgs,
        n_in_channels,
        deformable_group,
        out_h,
        out_w,
        use_mask,
        columns);
  } else {
    deformable_im2col_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
        num_kernels,
        input,
        offset,
        mask,
        height,
        width,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        parallel_imgs,
        n_in_channels,
        deformable_group,
        out_h,
        out_w,
        use_mask,
        columns);
  }
}

void deformable_im2col(
    const Tensor& input,
    const Tensor& data_offset,
    const Tensor& data_mask,
    int n_in_channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int out_h,
    int out_w,
    int parallel_imgs,
    int deformable_group,
    bool use_mask,
    Tensor data_col) {
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());

  const int64_t num_kernels =
      (int64_t)n_in_channels * out_h * out_w * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      input.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  // Checks if we should use 64bits indexing
  // https://github.com/pytorch/vision/issues/4269
  bool use_64bits_indexing = false;
  // Checks if num_kernels or columns numel larger than 2 ** 31
  use_64bits_indexing |= num_kernels > std::numeric_limits<int32_t>::max();
  use_64bits_indexing |=
      ((int64_t)n_in_channels * weight_h * weight_w * parallel_imgs * out_h *
           out_w >
       std::numeric_limits<int32_t>::max());

  THO_DISPATCH_V2(
      input.scalar_type(),
      "deformable_im2col",
      AT_WRAP([&]() {
        launch_deformable_im2col_kernel<scalar_t>(
            use_64bits_indexing,
            blocks,
            threads,
            stream,
            num_kernels,
            input.const_data_ptr<scalar_t>(),
            data_offset.const_data_ptr<scalar_t>(),
            data_mask.const_data_ptr<scalar_t>(),
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            n_in_channels,
            deformable_group,
            out_h,
            out_w,
            use_mask,
            data_col.mutable_data_ptr<scalar_t>());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

int get_greatest_divisor_below_bound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

template <typename scalar_t, typename index_t>
__global__ void deformable_col2im_kernel(
    index_t n,
    const scalar_t* col,
    const scalar_t* offset_ptr,
    const scalar_t* mask_ptr,
    index_t channels,
    index_t height,
    index_t width,
    index_t kernel_h,
    index_t kernel_w,
    index_t pad_h,
    index_t pad_w,
    index_t stride_h,
    index_t stride_w,
    index_t dilation_h,
    index_t dilation_w,
    index_t batch_sz,
    index_t n_offset_grps,
    index_t out_h,
    index_t out_w,
    bool use_mask,
    scalar_t* grad_im) {
  const index_t grad_im_numel = width * height * channels * batch_sz;

  CUDA_1D_KERNEL_LOOP_T(index, n, int64_t) {
    const index_t out_x = index % out_w;
    const index_t out_y = (index / out_w) % out_h;
    const index_t b = (index / (out_w * out_h)) % batch_sz;
    const index_t j = (index / (out_w * out_h * batch_sz)) % kernel_w;
    const index_t i =
        (index / (out_w * out_h * batch_sz * kernel_w)) % kernel_h;
    const index_t c = index / (out_w * out_h * batch_sz * kernel_w * kernel_h);

    index_t c_per_offset_grp = channels / n_offset_grps;
    const index_t offset_grp = c / c_per_offset_grp;

    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * kernel_h * kernel_w *
        out_h * out_w;

    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * kernel_h * kernel_w *
          out_h * out_w;
    }

    const index_t mask_idx = i * kernel_w + j;
    const index_t offset_idx = 2 * mask_idx;

    const index_t offset_h_ptr = ((offset_idx)*out_h + out_y) * out_w + out_x;
    const index_t offset_w_ptr =
        ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

    const scalar_t offset_h = offset_ptr[offset_h_ptr];
    const scalar_t offset_w = offset_ptr[offset_w_ptr];

    scalar_t mask_value = 1;
    if (use_mask) {
      mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
    }

    const scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
    const scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

    for (index_t dy = -1; dy <= 1; dy++) {
      for (index_t dx = -1; dx <= 1; dx++) {
        index_t yp = (index_t)y + dy;
        index_t xp = (index_t)x + dx;
        if (0 <= yp && yp < height && 0 <= xp && xp < width &&
            std::abs(y - yp) < 1 && std::abs(x - xp) < 1) {
          index_t grad_pos = ((b * channels + c) * height + yp) * width + xp;
          scalar_t weight = (1 - std::abs(y - yp)) * (1 - std::abs(x - xp));
          torch::headeronly::fastAtomicAdd(
              grad_im,
              grad_pos,
              grad_im_numel,
              mask_value * weight * col[index],
              true);
        }
      }
    }
  }
}

template <typename scalar_t>
void launch_deformable_col2im_kernel(
    bool use_64bits_indexing,
    unsigned int blocks,
    unsigned int threads,
    cudaStream_t stream,
    int64_t num_kernels,
    const scalar_t* columns,
    const scalar_t* offset,
    const scalar_t* mask,
    int channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int parallel_imgs,
    int n_offset_grps,
    int out_h,
    int out_w,
    bool use_mask,
    scalar_t* grad_im) {
  if (use_64bits_indexing) {
    deformable_col2im_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
        num_kernels,
        columns,
        offset,
        mask,
        channels,
        height,
        width,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        parallel_imgs,
        n_offset_grps,
        out_h,
        out_w,
        use_mask,
        grad_im);
  } else {
    deformable_col2im_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
        num_kernels,
        columns,
        offset,
        mask,
        channels,
        height,
        width,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        parallel_imgs,
        n_offset_grps,
        out_h,
        out_w,
        use_mask,
        grad_im);
  }
}

void compute_grad_input(
    const Tensor& columns,
    const Tensor& offset,
    const Tensor& mask,
    int channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int parallel_imgs,
    int n_offset_grps,
    bool use_mask,
    Tensor grad_im) {
  torch::stable::accelerator::DeviceGuard device_guard(
      columns.get_device_index());

  const int out_h =
      (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  const int out_w =
      (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

  const int64_t num_kernels =
      (int64_t)channels * weight_h * weight_w * out_h * out_w * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      columns.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  // Checks if we should use 64bits indexing
  // https://github.com/pytorch/vision/issues/4269
  bool use_64bits_indexing = false;
  // Checks if num_kernels or columns numel larger than 2 ** 31
  use_64bits_indexing |= num_kernels > std::numeric_limits<int32_t>::max();

  THO_DISPATCH_V2(
      columns.scalar_type(),
      "compute_grad_input",
      AT_WRAP([&]() {
        launch_deformable_col2im_kernel<scalar_t>(
            use_64bits_indexing,
            blocks,
            threads,
            stream,
            num_kernels,
            columns.const_data_ptr<scalar_t>(),
            offset.const_data_ptr<scalar_t>(),
            mask.const_data_ptr<scalar_t>(),
            channels,
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            n_offset_grps,
            out_h,
            out_w,
            use_mask,
            grad_im.mutable_data_ptr<scalar_t>());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename index_t>
__device__ scalar_t get_coordinate_weight(
    const scalar_t* im_data,
    index_t height,
    index_t width,
    scalar_t y,
    scalar_t x,
    bool is_y_direction) {
  index_t y_l = floor(y);
  index_t x_l = floor(x);
  index_t y_h = y_l + 1;
  index_t x_h = x_l + 1;

  bool valid_y_l = 0 <= y_l && y_l < height;
  bool valid_y_h = 0 <= y_h && y_h < height;
  bool valid_x_l = 0 <= x_l && x_l < width;
  bool valid_x_h = 0 <= x_h && x_h < width;

  scalar_t zero = 0;
  scalar_t v_yx = (valid_y_l && valid_x_l) ? im_data[y_l * width + x_l] : zero;
  scalar_t v_yX = (valid_y_l && valid_x_h) ? im_data[y_l * width + x_h] : zero;
  scalar_t v_Yx = (valid_y_h && valid_x_l) ? im_data[y_h * width + x_l] : zero;
  scalar_t v_YX = (valid_y_h && valid_x_h) ? im_data[y_h * width + x_h] : zero;

  if (is_y_direction) {
    scalar_t dx = x - x_l;
    return dx * (v_YX - v_yX) + (1 - dx) * (v_Yx - v_yx);
  } else {
    scalar_t dy = y - y_l;
    return dy * (v_YX - v_Yx) + (1 - dy) * (v_yX - v_yx);
  }
}

template <typename scalar_t, typename index_t>
__global__ void deformable_col2im_coord_kernel(
    index_t n,
    const scalar_t* col_ptr,
    const scalar_t* im_ptr,
    const scalar_t* offset_ptr,
    const scalar_t* mask_ptr,
    index_t channels,
    index_t height,
    index_t width,
    index_t weight_h,
    index_t weight_w,
    index_t pad_h,
    index_t pad_w,
    index_t stride_h,
    index_t stride_w,
    index_t dilation_h,
    index_t dilation_w,
    index_t batch_sz,
    index_t offset_channels,
    index_t n_offset_grps,
    index_t out_h,
    index_t out_w,
    const bool use_mask,
    scalar_t* grad_offset,
    scalar_t* grad_mask) {
  CUDA_1D_KERNEL_LOOP_T(index, n, int64_t) {
    scalar_t grad_offset_val = 0;
    scalar_t grad_mask_val = 0;

    index_t w = index % out_w;
    index_t h = (index / out_w) % out_h;
    index_t w_w = (index / (out_w * out_h * 2)) % weight_w;
    index_t w_h = (index / (out_w * out_h * 2 * weight_w)) % weight_h;
    index_t c = (index / (out_w * out_h)) % offset_channels;
    index_t b = index / (out_w * out_h * offset_channels);

    const index_t offset_grp = c / (2 * weight_h * weight_w);
    const index_t col_step = weight_h * weight_w;

    index_t c_per_offset_grp = channels / n_offset_grps;

    col_ptr += offset_grp * c_per_offset_grp * weight_h * weight_w * batch_sz *
        out_w * out_h;
    im_ptr +=
        (b * n_offset_grps + offset_grp) * c_per_offset_grp * height * width;
    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * weight_h * weight_w *
        out_h * out_w;

    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * weight_h * weight_w *
          out_h * out_w;
    }

    const index_t offset_c = c - offset_grp * 2 * weight_h * weight_w;
    const bool is_y_direction = offset_c % 2 == 0;

    const index_t c_bound = c_per_offset_grp * weight_h * weight_w;
    for (index_t col_c = (offset_c / 2); col_c < c_bound; col_c += col_step) {
      const index_t col_pos =
          (((col_c * batch_sz + b) * out_h) + h) * out_w + w;

      index_t out_x = col_pos % out_w;
      index_t out_y = (col_pos / out_w) % out_h;
      index_t j = (col_pos / (out_w * out_h * batch_sz)) % weight_w;
      index_t i = (col_pos / (out_w * out_h * batch_sz * weight_w)) % weight_h;

      const index_t mask_idx = i * weight_w + j;

      const index_t offset_h_ptr =
          (((2 * mask_idx) * out_h + out_y) * out_w + out_x);
      const index_t offset_w_ptr =
          (((2 * mask_idx + 1) * out_h + out_y) * out_w + out_x);
      const scalar_t offset_h = offset_ptr[offset_h_ptr];
      const scalar_t offset_w = offset_ptr[offset_w_ptr];

      scalar_t mask_value = 1;
      if (use_mask) {
        mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
      }

      scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
      scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

      const scalar_t weight =
          get_coordinate_weight(im_ptr, height, width, y, x, is_y_direction);
      grad_offset_val += mask_value * weight * col_ptr[col_pos];

      if (use_mask && is_y_direction) {
        grad_mask_val += col_ptr[col_pos] *
            bilinear_interpolate(im_ptr, height, width, y, x);
      }

      im_ptr += height * width;
    }

    grad_offset[index] = grad_offset_val;

    if (use_mask && is_y_direction) {
      const index_t idx =
          ((((b * n_offset_grps + offset_grp) * weight_h + w_h) * weight_w +
            w_w) *
               out_h +
           h) *
              out_w +
          w;
      grad_mask[idx] = grad_mask_val;
    }
  }
}

template <typename scalar_t>
void launch_deformable_col2im_coord_kernel(
    bool use_64bits_indexing,
    unsigned int blocks,
    unsigned int threads,
    cudaStream_t stream,
    int64_t num_kernels,
    const scalar_t* columns,
    const scalar_t* input,
    const scalar_t* offset,
    const scalar_t* mask,
    int channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int parallel_imgs,
    int offset_channels,
    int n_offset_grps,
    int out_h,
    int out_w,
    bool use_mask,
    scalar_t* grad_offset,
    scalar_t* grad_mask) {
  if (use_64bits_indexing) {
    deformable_col2im_coord_kernel<scalar_t, int64_t>
        <<<blocks, threads, 0, stream>>>(
            num_kernels,
            columns,
            input,
            offset,
            mask,
            channels,
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            offset_channels,
            n_offset_grps,
            out_h,
            out_w,
            use_mask,
            grad_offset,
            grad_mask);
  } else {
    deformable_col2im_coord_kernel<scalar_t, int>
        <<<blocks, threads, 0, stream>>>(
            num_kernels,
            columns,
            input,
            offset,
            mask,
            channels,
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            offset_channels,
            n_offset_grps,
            out_h,
            out_w,
            use_mask,
            grad_offset,
            grad_mask);
  }
}

void compute_grad_offset_and_mask(
    const Tensor& columns,
    const Tensor& input,
    const Tensor& offset,
    const Tensor& mask,
    int channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int parallel_imgs,
    int n_offset_grps,
    bool use_mask,
    Tensor grad_offset,
    Tensor grad_mask) {
  torch::stable::accelerator::DeviceGuard device_guard(
      columns.get_device_index());

  const int out_h =
      (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  const int out_w =
      (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  const int64_t num_kernels = (int64_t)out_h * out_w * 2 * weight_h * weight_w *
      n_offset_grps * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      columns.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  // Checks if we should use 64bits indexing
  // https://github.com/pytorch/vision/issues/4269
  bool use_64bits_indexing = false;
  // Checks if columns numel is larger than 2 ** 31
  use_64bits_indexing |= num_kernels > std::numeric_limits<int32_t>::max();
  use_64bits_indexing |=
      ((int64_t)channels * weight_h * weight_w * parallel_imgs * out_h * out_w >
       std::numeric_limits<int32_t>::max());

  THO_DISPATCH_V2(
      columns.scalar_type(),
      "compute_grad_offset_and_mask",
      AT_WRAP([&]() {
        launch_deformable_col2im_coord_kernel<scalar_t>(
            use_64bits_indexing,
            blocks,
            threads,
            stream,
            num_kernels,
            columns.const_data_ptr<scalar_t>(),
            input.const_data_ptr<scalar_t>(),
            offset.const_data_ptr<scalar_t>(),
            mask.const_data_ptr<scalar_t>(),
            channels,
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            2 * weight_h * weight_w * n_offset_grps,
            n_offset_grps,
            out_h,
            out_w,
            use_mask,
            grad_offset.mutable_data_ptr<scalar_t>(),
            grad_mask.mutable_data_ptr<scalar_t>());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<Tensor, Tensor, Tensor> backward_gradient_inputs(
    Tensor input,
    Tensor weight,
    Tensor offset,
    Tensor mask,
    Tensor grad_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int n_weight_grps,
    int n_offset_grps,
    int n_parallel_imgs,
    bool use_mask) {
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());

  int batch_sz = input.size(0);
  long n_in_channels = input.size(1);
  long in_h = input.size(2);
  long in_w = input.size(3);

  n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);

  long n_out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  long out_w =
      (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  long out_h =
      (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;

  auto grad_input = torch::stable::new_zeros(input, input.sizes());
  auto grad_offset = torch::stable::new_zeros(offset, offset.sizes());
  auto grad_mask = torch::stable::new_zeros(mask, mask.sizes());

  if (batch_sz == 0) {
    return std::make_tuple(grad_input, grad_offset, grad_mask);
  }

  auto columns = torch::stable::new_empty(
      input,
      {n_in_channels * weight_w * weight_h, n_parallel_imgs * out_h * out_w});

  // Separate into blocks
  grad_input = torch::stable::reshape(
      grad_input,
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});
  input = torch::stable::reshape(
      input,
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  grad_offset = torch::stable::reshape(
      grad_offset,
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});
  offset = torch::stable::reshape(
      offset,
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});

  if (use_mask) {
    grad_mask = torch::stable::reshape(
        grad_mask,
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
    mask = torch::stable::reshape(
        mask,
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
  }

  grad_out = torch::stable::permute(
      torch::stable::reshape(
          grad_out,
          {batch_sz / n_parallel_imgs,
           n_parallel_imgs,
           n_weight_grps,
           n_out_channels / n_weight_grps,
           out_h,
           out_w}),
      {0, 2, 3, 1, 4, 5});

  weight = torch::stable::reshape(
      weight,
      {n_weight_grps,
       weight.size(0) / n_weight_grps,
       weight.size(1),
       weight.size(2),
       weight.size(3)});

  columns = torch::stable::view(
      columns,
      {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});

  for (int elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
    torch::stable::zero_(columns);
    // Separate into weight groups
    for (int g = 0; g < n_weight_grps; g++) {
      auto columns_g = torch::stable::select(columns, 0, g);
      auto weight_g = torch::stable::transpose(
          torch::stable::flatten(torch::stable::select(weight, 0, g), 1), 0, 1);
      auto grad_out_g = torch::stable::flatten(
          torch::stable::select(torch::stable::select(grad_out, 0, elt), 0, g),
          1);
      stable_helpers::mm_out(weight_g, grad_out_g, columns_g);
    }

    compute_grad_offset_and_mask(
        columns,
        torch::stable::select(input, 0, elt),
        torch::stable::select(offset, 0, elt),
        torch::stable::select(mask, 0, elt),
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        torch::stable::select(grad_offset, 0, elt),
        torch::stable::select(grad_mask, 0, elt));

    compute_grad_input(
        columns,
        torch::stable::select(offset, 0, elt),
        torch::stable::select(mask, 0, elt),
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        torch::stable::select(grad_input, 0, elt));
  }

  grad_input =
      torch::stable::view(grad_input, {batch_sz, n_in_channels, in_h, in_w});
  grad_offset = torch::stable::view(
      grad_offset,
      {batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  if (use_mask) {
    grad_mask = torch::stable::view(
        grad_mask,
        {batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w});
  }

  return std::make_tuple(grad_input, grad_offset, grad_mask);
}

Tensor backward_gradient_parameters(
    Tensor input,
    const Tensor& weight,
    Tensor offset,
    Tensor mask,
    const Tensor& grad_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int n_weight_grps,
    int n_offset_grps,
    int n_parallel_imgs,
    bool use_mask) {
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());

  int batch_sz = input.size(0);
  long n_in_channels = input.size(1);
  long in_h = input.size(2);
  long in_w = input.size(3);

  n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);

  long n_out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  long out_h = grad_out.size(2);
  long out_w = grad_out.size(3);

  auto grad_weight = torch::stable::new_zeros(weight, weight.sizes());
  if (batch_sz == 0) {
    return grad_weight;
  }

  Tensor grad_out_buf = torch::stable::contiguous(torch::stable::permute(
      torch::stable::reshape(
          grad_out,
          {batch_sz / n_parallel_imgs,
           n_parallel_imgs,
           n_weight_grps,
           n_out_channels / n_weight_grps,
           out_h,
           out_w}),
      {0, 2, 3, 1, 4, 5}));

  input = torch::stable::reshape(
      input,
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  offset = torch::stable::reshape(
      offset,
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});

  if (use_mask) {
    mask = torch::stable::reshape(
        mask,
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
  }

  grad_weight = torch::stable::view(
      grad_weight,
      {n_weight_grps,
       grad_weight.size(0) / n_weight_grps,
       grad_weight.size(1),
       grad_weight.size(2),
       grad_weight.size(3)});

  auto columns = torch::stable::new_empty(
      input,
      {n_weight_grps,
       n_in_channels * weight_w * weight_h / n_weight_grps,
       n_parallel_imgs * out_h * out_w});

  for (int elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
    deformable_im2col(
        torch::stable::select(input, 0, elt),
        torch::stable::select(offset, 0, elt),
        torch::stable::select(mask, 0, elt),
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        out_h,
        out_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        columns);

    for (int g = 0; g < n_weight_grps; g++) {
      auto grad_weight_g =
          torch::stable::flatten(torch::stable::select(grad_weight, 0, g), 1);
      auto grad_out_buf_g = torch::stable::flatten(
          torch::stable::select(
              torch::stable::select(grad_out_buf, 0, elt), 0, g),
          1);
      auto columns_g =
          torch::stable::transpose(torch::stable::select(columns, 0, g), 1, 0);
      auto update = torch::stable::matmul(grad_out_buf_g, columns_g);
      torch::stable::copy_(grad_weight_g, add_tensors(grad_weight_g, update));
    }
  }

  grad_weight = torch::stable::view(
      grad_weight,
      {grad_weight.size(0) * grad_weight.size(1),
       grad_weight.size(2),
       grad_weight.size(3),
       grad_weight.size(4)});
  return grad_weight;
}

Tensor deform_conv2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& mask,
    const Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask) {
  Tensor input_c = torch::stable::contiguous(input);
  Tensor offset_c = torch::stable::contiguous(offset);
  Tensor weight_c = torch::stable::contiguous(weight);
  Tensor mask_c = torch::stable::contiguous(mask);
  Tensor bias_c = torch::stable::contiguous(bias);

  STD_TORCH_CHECK(input_c.dim() == 4);
  STD_TORCH_CHECK(offset_c.dim() == 4);
  STD_TORCH_CHECK(!use_mask || mask_c.dim() == 4);
  STD_TORCH_CHECK(weight_c.dim() == 4);
  STD_TORCH_CHECK(input_c.is_cuda(), "input must be a CUDA tensor");

  torch::stable::accelerator::DeviceGuard device_guard(
      input_c.get_device_index());

  int batch_sz = input_c.size(0);
  int in_channels = input_c.size(1);
  int in_h = input_c.size(2);
  int in_w = input_c.size(3);

  int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  // Unpack shapes and args
  int out_channels = weight_c.size(0);
  int weight_h = weight_c.size(2);
  int weight_w = weight_c.size(3);

  int ker_h = dilation_h * (weight_h - 1) + 1;
  int ker_w = dilation_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  STD_TORCH_CHECK(
      weight_h > 0 && weight_w > 0,
      "weight_h: ",
      weight_h,
      " weight_w: ",
      weight_w);
  STD_TORCH_CHECK(
      stride_h > 0 && stride_w > 0,
      "stride_h: ",
      stride_h,
      " stride_w: ",
      stride_w);
  STD_TORCH_CHECK(
      pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_h, " pad_w: ", pad_w);
  STD_TORCH_CHECK(
      dilation_h > 0 && dilation_w > 0,
      "dilation_h: ",
      dilation_h,
      " dilation_w: ",
      dilation_w);

  STD_TORCH_CHECK(weight_c.size(1) * n_weight_grps == input_c.size(1));
  STD_TORCH_CHECK(weight_c.size(0) % n_weight_grps == 0);
  STD_TORCH_CHECK(
      (offset_c.size(1) == n_offset_grps * 2 * weight_h * weight_w),
      "offset.shape[1] is not valid: got: ",
      offset_c.size(1),
      " expected: ",
      n_offset_grps * 2 * weight_h * weight_w);
  STD_TORCH_CHECK(
      (!use_mask || mask_c.size(1) == n_offset_grps * weight_h * weight_w),
      "mask.shape[1] is not valid: got: ",
      mask_c.size(1),
      " expected: ",
      n_offset_grps * weight_h * weight_w);
  STD_TORCH_CHECK(input_c.size(1) % n_offset_grps == 0);

  STD_TORCH_CHECK(
      (offset_c.size(0) == input_c.size(0)), "invalid batch size of offset");
  STD_TORCH_CHECK(
      (offset_c.size(2) == out_h && offset_c.size(3) == out_w),
      "offset output dims: (",
      offset_c.size(2),
      ", ",
      offset_c.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  STD_TORCH_CHECK(
      (mask_c.size(0) == input_c.size(0)), "invalid batch size of mask");
  STD_TORCH_CHECK(
      (!use_mask || (mask_c.size(2) == out_h && mask_c.size(3) == out_w)),
      "mask output dims: (",
      mask_c.size(2),
      ", ",
      mask_c.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  STD_TORCH_CHECK(
      out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ",
      out_h,
      " out_w: ",
      out_w);

  auto out =
      torch::stable::new_zeros(input_c, {batch_sz, out_channels, out_h, out_w});
  if (batch_sz == 0) {
    return out;
  }

  // Separate batches into blocks
  out = torch::stable::view(
      out,
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       out_channels,
       out_h,
       out_w});
  input_c = torch::stable::view(
      input_c,
      {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_h, in_w});

  offset_c = torch::stable::view(
      offset_c,
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});

  if (use_mask) {
    mask_c = torch::stable::view(
        mask_c,
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
  }

  Tensor out_buf = torch::stable::new_zeros(
      out,
      {batch_sz / n_parallel_imgs,
       out_channels,
       n_parallel_imgs * out_h,
       out_w});

  // Separate channels into convolution groups
  out_buf = torch::stable::view(
      out_buf,
      {out_buf.size(0),
       n_weight_grps,
       out_buf.size(1) / n_weight_grps,
       out_buf.size(2),
       out_buf.size(3)});
  weight_c = torch::stable::view(
      weight_c,
      {n_weight_grps,
       weight_c.size(0) / n_weight_grps,
       weight_c.size(1),
       weight_c.size(2),
       weight_c.size(3)});

  // Sample points and perform convolution
  auto columns = torch::stable::new_zeros(
      input_c,
      {in_channels * weight_h * weight_w, n_parallel_imgs * out_h * out_w});
  for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
    deformable_im2col(
        torch::stable::select(input_c, 0, b),
        torch::stable::select(offset_c, 0, b),
        torch::stable::select(mask_c, 0, b),
        in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        out_h,
        out_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        columns);

    columns = torch::stable::view(
        columns,
        {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int g = 0; g < n_weight_grps; g++) {
      auto out_buf_g = torch::stable::flatten(
          torch::stable::select(torch::stable::select(out_buf, 0, b), 0, g), 1);
      auto weight_g =
          torch::stable::flatten(torch::stable::select(weight_c, 0, g), 1);
      auto columns_g = torch::stable::select(columns, 0, g);
      stable_helpers::mm_out(weight_g, columns_g, out_buf_g);
    }
    columns = torch::stable::view(
        columns, {columns.size(0) * columns.size(1), columns.size(2)});
  }

  out_buf = torch::stable::view(
      out_buf,
      {batch_sz / n_parallel_imgs,
       out_channels,
       n_parallel_imgs,
       out_h,
       out_w});
  out_buf = torch::stable::transpose(out_buf, 1, 2);
  torch::stable::copy_(out, out_buf);
  out = torch::stable::view(out, {batch_sz, out_channels, out_h, out_w});

  return add_tensors(out, torch::stable::view(bias_c, {1, out_channels, 1, 1}));
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> deform_conv2d_backward_kernel(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& mask,
    const Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask) {
  Tensor grad_out_c = torch::stable::contiguous(grad_out);
  Tensor input_c = torch::stable::contiguous(input);
  Tensor weight_c = torch::stable::contiguous(weight);
  Tensor offset_c = torch::stable::contiguous(offset);
  Tensor mask_c = torch::stable::contiguous(mask);
  Tensor bias_c = torch::stable::contiguous(bias);

  const int batch_sz = input_c.size(0);
  const int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  auto grad_input_and_offset_and_mask = backward_gradient_inputs(
      input_c,
      weight_c,
      offset_c,
      mask_c,
      grad_out_c,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      n_weight_grps,
      n_offset_grps,
      n_parallel_imgs,
      use_mask);

  auto grad_input = std::get<0>(grad_input_and_offset_and_mask);
  auto grad_offset = std::get<1>(grad_input_and_offset_and_mask);
  auto grad_mask = std::get<2>(grad_input_and_offset_and_mask);

  auto grad_weight = backward_gradient_parameters(
      input_c,
      weight_c,
      offset_c,
      mask_c,
      grad_out_c,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      n_weight_grps,
      n_offset_grps,
      n_parallel_imgs,
      use_mask);

  std::array<int64_t, 3> bias_sum_dims{0, 2, 3};
  auto grad_out_sum = torch::stable::sum(
      grad_out_c, torch::headeronly::IntHeaderOnlyArrayRef(bias_sum_dims));
  auto grad_bias = add_tensors(
      torch::stable::new_zeros(bias_c, bias_c.sizes()), grad_out_sum);

  return std::make_tuple(
      grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("deform_conv2d", TORCH_BOX(&deform_conv2d_forward_kernel));
  m.impl("_deform_conv2d_backward", TORCH_BOX(&deform_conv2d_backward_kernel));
}

} // namespace ops
} // namespace vision
