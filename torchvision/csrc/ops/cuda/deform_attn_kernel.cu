// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This software may be used and distributed in accordance with
// the terms of the DINOv3 License Agreement.

/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <THC/THCAtomics.cuh>

namespace vision {
namespace ops {

namespace {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t*& bottom_data,
    const int& height,
    const int& width,
    const int& nheads,
    const int& channels,
    const scalar_t& h,
    const scalar_t& w,
    const int& m,
    const int& c) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(
    const scalar_t*& bottom_data,
    const int& height,
    const int& width,
    const int& nheads,
    const int& channels,
    const scalar_t& h,
    const scalar_t& w,
    const int& m,
    const int& c,
    const scalar_t& top_grad,
    const scalar_t& attn_weight,
    scalar_t*& grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(
    const scalar_t*& bottom_data,
    const int& height,
    const int& width,
    const int& nheads,
    const int& channels,
    const scalar_t& h,
    const scalar_t& w,
    const int& m,
    const int& c,
    const scalar_t& top_grad,
    const scalar_t& attn_weight,
    scalar_t*& grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val);
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t* data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t* data_value_ptr = data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += ms_deform_attn_im2col_bilinear(
                     data_value_ptr,
                     spatial_h,
                     spatial_w,
                     num_heads,
                     channels,
                     h_im,
                     w_im,
                     m_col,
                     c_col) *
              weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t* data_value_ptr = data_value + value_ptr_offset;
      scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr,
              spatial_h,
              spatial_w,
              num_heads,
              channels,
              h_im,
              w_im,
              m_col,
              c_col,
              top_grad,
              weight,
              grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int tid = 1; tid < blockSize; ++tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }

          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t* data_value_ptr = data_value + value_ptr_offset;
      scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr,
              spatial_h,
              spatial_w,
              num_heads,
              channels,
              h_im,
              w_im,
              m_col,
              c_col,
              top_grad,
              weight,
              grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t* data_value_ptr = data_value + value_ptr_offset;
      scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr,
              spatial_h,
              spatial_w,
              num_heads,
              channels,
              h_im,
              w_im,
              m_col,
              c_col,
              top_grad,
              weight,
              grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int tid = 1; tid < blockDim.x; ++tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }

          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t* data_value_ptr = data_value + value_ptr_offset;
      scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr,
              spatial_h,
              spatial_w,
              num_heads,
              channels,
              h_im,
              w_im,
              m_col,
              c_col,
              top_grad,
              weight,
              grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t* data_value_ptr = data_value + value_ptr_offset;
      scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr,
              spatial_h,
              spatial_w,
              num_heads,
              channels,
              h_im,
              w_im,
              m_col,
              c_col,
              top_grad,
              weight,
              grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_attn_weight, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(
    const int n,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t* data_value_ptr = data_value + value_ptr_offset;
      scalar_t* grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear_gm(
              data_value_ptr,
              spatial_h,
              spatial_w,
              num_heads,
              channels,
              h_im,
              w_im,
              m_col,
              c_col,
              top_grad,
              weight,
              grad_value_ptr,
              grad_sampling_loc,
              grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
          num_kernels,
          data_value,
          data_spatial_shapes,
          data_level_start_index,
          data_sampling_loc,
          data_attn_weight,
          batch_size,
          spatial_size,
          num_heads,
          channels,
          num_levels,
          num_query,
          num_point,
          data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void ms_deformable_col2im_cuda(
    cudaStream_t stream,
    const scalar_t* grad_col,
    const scalar_t* data_value,
    const int64_t* data_spatial_shapes,
    const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc,
    const scalar_t* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* grad_value,
    scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
  const int num_threads =
      (channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > 1024) {
    if ((channels & 1023) == 0) {
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads),
             num_threads,
             num_threads * 3 * sizeof(scalar_t),
             stream>>>(
              num_kernels,
              grad_col,
              data_value,
              data_spatial_shapes,
              data_level_start_index,
              data_sampling_loc,
              data_attn_weight,
              batch_size,
              spatial_size,
              num_heads,
              channels,
              num_levels,
              num_query,
              num_point,
              grad_value,
              grad_sampling_loc,
              grad_attn_weight);
    } else {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads),
             num_threads,
             0,
             stream>>>(
              num_kernels,
              grad_col,
              data_value,
              data_spatial_shapes,
              data_level_start_index,
              data_sampling_loc,
              data_attn_weight,
              batch_size,
              spatial_size,
              num_heads,
              channels,
              num_levels,
              num_query,
              num_point,
              grad_value,
              grad_sampling_loc,
              grad_attn_weight);
    }
  } else {
    switch (channels) {
      case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<
            scalar_t,
            1>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<
            scalar_t,
            2>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<
            scalar_t,
            4>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<
            scalar_t,
            8>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<
            scalar_t,
            16>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 32:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<
            scalar_t,
            32>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<
            scalar_t,
            64>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<
            scalar_t,
            128>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<
            scalar_t,
            256>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<
            scalar_t,
            512>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<
            scalar_t,
            1024>
            <<<GET_BLOCKS(num_actual_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                grad_col,
                data_value,
                data_spatial_shapes,
                data_level_start_index,
                data_sampling_loc,
                data_attn_weight,
                batch_size,
                spatial_size,
                num_heads,
                channels,
                num_levels,
                num_query,
                num_point,
                grad_value,
                grad_sampling_loc,
                grad_attn_weight);
        break;
      default:
        if (channels < 64) {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads),
                 num_threads,
                 num_threads * 3 * sizeof(scalar_t),
                 stream>>>(
                  num_kernels,
                  grad_col,
                  data_value,
                  data_spatial_shapes,
                  data_level_start_index,
                  data_sampling_loc,
                  data_attn_weight,
                  batch_size,
                  spatial_size,
                  num_heads,
                  channels,
                  num_levels,
                  num_query,
                  num_point,
                  grad_value,
                  grad_sampling_loc,
                  grad_attn_weight);
        } else {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads),
                 num_threads,
                 num_threads * 3 * sizeof(scalar_t),
                 stream>>>(
                  num_kernels,
                  grad_col,
                  data_value,
                  data_spatial_shapes,
                  data_level_start_index,
                  data_sampling_loc,
                  data_attn_weight,
                  batch_size,
                  spatial_size,
                  num_heads,
                  channels,
                  num_levels,
                  num_query,
                  num_point,
                  grad_value,
                  grad_sampling_loc,
                  grad_attn_weight);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

at::Tensor ms_deform_attn_forward_kernel(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const int64_t im2col_step) {
  // Basic tensor properties
  TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
  TORCH_CHECK(
      spatial_shapes.is_contiguous(),
      "spatial_shapes tensor has to be contiguous");
  TORCH_CHECK(
      level_start_index.is_contiguous(),
      "level_start_index tensor has to be contiguous");
  TORCH_CHECK(
      sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
  TORCH_CHECK(
      attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  TORCH_CHECK(
      level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
  TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
  TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

  // Rank checks
  TORCH_CHECK(
      value.dim() == 4, "value must be a 4D tensor of shape [B, Nv, H, C]");
  TORCH_CHECK(
      spatial_shapes.dim() == 2 && spatial_shapes.size(1) == 2,
      "spatial_shapes must be a 2D tensor of shape [L, 2]");
  TORCH_CHECK(
      level_start_index.dim() == 1,
      "level_start_index must be a 1D tensor of shape [L]");
  TORCH_CHECK(
      sampling_loc.dim() == 6,
      "sampling_loc must be a 6D tensor of shape [B, Nq, H, L, P, 2]");
  TORCH_CHECK(
      sampling_loc.size(5) == 2,
      "sampling_loc last dimension must be 2 (normalized x, y)");
  TORCH_CHECK(
      attn_weight.dim() == 5,
      "attn_weight must be a 5D tensor of shape [B, Nq, H, L, P]");

  const auto batch = value.size(0);
  const auto spatial_size = value.size(1);
  const auto num_heads = value.size(2);
  const auto channels = value.size(3);

  const auto num_levels = spatial_shapes.size(0);

  const auto num_query = sampling_loc.size(1);
  const auto num_point = sampling_loc.size(4);

  // Cross-shape consistency checks
  TORCH_CHECK(
      batch == sampling_loc.size(0),
      "batch size mismatch between value (B) and sampling_loc (B)");
  TORCH_CHECK(
      num_heads == sampling_loc.size(2),
      "num_heads mismatch between value (H) and sampling_loc (H)");
  TORCH_CHECK(
      num_heads == attn_weight.size(2),
      "num_heads mismatch between value (H) and attn_weight (H)");

  TORCH_CHECK(
      num_levels == sampling_loc.size(3),
      "L mismatch: spatial_shapes.size(0) must equal sampling_loc.size(3)");
  TORCH_CHECK(
      level_start_index.size(0) == num_levels,
      "level_start_index.size(0) must equal spatial_shapes.size(0) (L)");

  TORCH_CHECK(
      attn_weight.size(0) == sampling_loc.size(0) &&
          attn_weight.size(1) == sampling_loc.size(1) &&
          attn_weight.size(2) == sampling_loc.size(2) &&
          attn_weight.size(3) == sampling_loc.size(3) &&
          attn_weight.size(4) == sampling_loc.size(4),
      "attn_weight must match sampling_loc in [B, Nq, H, L, P]");

  // Nv must equal sum_{l}(H_l * W_l) from spatial_shapes
  const auto expected_spatial_size =
      (spatial_shapes.prod(1)).sum().item<int64_t>();
  TORCH_CHECK(
      spatial_size == expected_spatial_size,
      "value.size(1) (Nv) must equal sum over levels of (H_l * W_l)");

  // im2col_step validation
  const auto im2col_step_ = std::min(batch, im2col_step);
  TORCH_CHECK(im2col_step_ > 0, "im2col_step must be > 0");
  TORCH_CHECK(
      batch % im2col_step_ == 0, "batch must be divisible by im2col_step");

  auto output =
      at::zeros({batch, num_query, num_heads, channels}, value.options());

  const auto batch_n = im2col_step_;
  auto output_n = output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});
  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto columns = output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
          ms_deformable_im2col_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              batch_n,
              spatial_size,
              num_heads,
              channels,
              num_levels,
              num_query,
              num_point,
              columns.data_ptr<scalar_t>());
        }));
  }

  output = output.view({batch, num_query, num_heads * channels});
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ms_deform_attn_backward_kernel(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    const int64_t im2col_step) {
  // Basic tensor properties
  TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
  TORCH_CHECK(
      spatial_shapes.is_contiguous(),
      "spatial_shapes tensor has to be contiguous");
  TORCH_CHECK(
      level_start_index.is_contiguous(),
      "level_start_index tensor has to be contiguous");
  TORCH_CHECK(
      sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
  TORCH_CHECK(
      attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
  TORCH_CHECK(
      grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  TORCH_CHECK(
      level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
  TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
  TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

  // Rank checks mirroring forward
  TORCH_CHECK(
      value.dim() == 4, "value must be a 4D tensor of shape [B, Nv, H, C]");
  TORCH_CHECK(
      spatial_shapes.dim() == 2 && spatial_shapes.size(1) == 2,
      "spatial_shapes must be a 2D tensor of shape [L, 2]");
  TORCH_CHECK(
      level_start_index.dim() == 1,
      "level_start_index must be a 1D tensor of shape [L]");
  TORCH_CHECK(
      sampling_loc.dim() == 6,
      "sampling_loc must be a 6D tensor of shape [B, Nq, H, L, P, 2]");
  TORCH_CHECK(
      sampling_loc.size(5) == 2,
      "sampling_loc last dimension must be 2 (normalized x, y)");
  TORCH_CHECK(
      attn_weight.dim() == 5,
      "attn_weight must be a 5D tensor of shape [B, Nq, H, L, P]");

  const auto batch = value.size(0);
  const auto spatial_size = value.size(1);
  const auto num_heads = value.size(2);
  const auto channels = value.size(3);

  const auto num_levels = spatial_shapes.size(0);

  const auto num_query = sampling_loc.size(1);
  const auto num_point = sampling_loc.size(4);

  // Cross-shape consistency checks
  TORCH_CHECK(
      batch == sampling_loc.size(0),
      "batch size mismatch between value (B) and sampling_loc (B)");
  TORCH_CHECK(
      num_heads == sampling_loc.size(2),
      "num_heads mismatch between value (H) and sampling_loc (H)");
  TORCH_CHECK(
      num_heads == attn_weight.size(2),
      "num_heads mismatch between value (H) and attn_weight (H)");
  TORCH_CHECK(
      num_levels == sampling_loc.size(3),
      "L mismatch: spatial_shapes.size(0) must equal sampling_loc.size(3)");
  TORCH_CHECK(
      level_start_index.size(0) == num_levels,
      "level_start_index.size(0) must equal spatial_shapes.size(0) (L)");
  TORCH_CHECK(
      attn_weight.size(0) == sampling_loc.size(0) &&
          attn_weight.size(1) == sampling_loc.size(1) &&
          attn_weight.size(2) == sampling_loc.size(2) &&
          attn_weight.size(3) == sampling_loc.size(3) &&
          attn_weight.size(4) == sampling_loc.size(4),
      "attn_weight must match sampling_loc in [B, Nq, H, L, P]");

  // Nv must equal sum_{l}(H_l * W_l) from spatial_shapes
  const auto expected_spatial_size =
      (spatial_shapes.prod(1)).sum().item<int64_t>();
  TORCH_CHECK(
      spatial_size == expected_spatial_size,
      "value.size(1) (Nv) must equal sum over levels of (H_l * W_l)");

  // grad_output should have the same total elements as [B, Nq, H, C]
  const auto expected_grad_elems = batch * num_query * num_heads * channels;
  TORCH_CHECK(
      grad_output.numel() == expected_grad_elems,
      "grad_output must contain B * Nq * H * C elements");

  // im2col_step validation
  const auto im2col_step_ = std::min(batch, im2col_step);
  TORCH_CHECK(im2col_step_ > 0, "im2col_step must be > 0");
  TORCH_CHECK(
      batch % im2col_step_ == 0, "batch must be divisible by im2col_step");

  auto grad_value = at::zeros_like(value);
  auto grad_sampling_loc = at::zeros_like(sampling_loc);
  auto grad_attn_weight = at::zeros_like(attn_weight);

  const auto batch_n = im2col_step_;
  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  auto grad_output_n = grad_output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto grad_output_g = grad_output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
          ms_deformable_col2im_cuda(
              at::cuda::getCurrentCUDAStream(),
              grad_output_g.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              batch_n,
              spatial_size,
              num_heads,
              channels,
              num_levels,
              num_query,
              num_point,
              grad_value.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_value_size,
              grad_sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              grad_attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size);
        }));
  }

  return {grad_value, grad_sampling_loc, grad_attn_weight};
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_attn"),
      TORCH_FN(ms_deform_attn_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_deform_attn_backward"),
      TORCH_FN(ms_deform_attn_backward_kernel));
}

} // namespace ops
} // namespace vision
