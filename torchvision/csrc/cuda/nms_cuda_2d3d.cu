/*
NMS implementation in CUDA from pytorch framework
(https://github.com/pytorch/vision/tree/master/torchvision/csrc/cuda on Nov 13 2019)

Adapted for additional 3D capability by G. Ramien, DKFZ Heidelberg
*/

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "cuda_helpers.h"

#include <iostream>
#include <vector>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline float devIoU(T const* const a, T const* const b) {
  // a, b hold box coords as (y1, x1, y2, x2) with y1 < y2 etc.
  T bottom = max(a[0], b[0]), top = min(a[2], b[2]);
  T left = max(a[1], b[1]), right = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(top - bottom, (T)0);
  T interS = width * height;

  T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]);

  return interS / (Sa + Sb - interS);
}

template <typename T>
__device__ inline float devIoU_3d(T const* const a, T const* const b) {
  // a, b hold box coords as (y1, x1, y2, x2, z1, z2) with y1 < y2 etc.
  // get coordinates of intersection, calc intersection
  T bottom = max(a[0], b[0]), top = min(a[2], b[2]);
  T left = max(a[1], b[1]), right = min(a[3], b[3]);
  T front = max(a[4], b[4]), back = min(a[5], b[5]);
  T width = max(right - left, (T)0), height = max(top - bottom, (T)0);
  T depth = max(back - front, (T)0);
  T interS = width * height * depth;
  // calc separate boxes volumes
  T Sa = (a[2] - a[0]) * (a[3] - a[1]) * (a[5] - a[4]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]) * (b[5] - b[4]);

  return interS / (Sa + Sb - interS);
}


template <typename T>
__global__ void nms_kernel(const int n_boxes, const float iou_threshold, const T* dev_boxes,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;
  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU<T>(cur_box, block_boxes + i * 4) > iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = at::cuda::ATenCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


template <typename T>
__global__ void nms_kernel_3d(const int n_boxes, const float iou_threshold, const T* dev_boxes,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;
  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 6];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 6 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 0];
    block_boxes[threadIdx.x * 6 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 1];
    block_boxes[threadIdx.x * 6 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 2];
    block_boxes[threadIdx.x * 6 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 3];
    block_boxes[threadIdx.x * 6 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 4];
    block_boxes[threadIdx.x * 6 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 5];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 6;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU_3d<T>(cur_box, block_boxes + i * 6) > iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = at::cuda::ATenCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


at::Tensor nms_cuda(const at::Tensor& dets, const at::Tensor& scores, float iou_threshold) {
  /* dets expected as (n_dets, dim) where dim=4 in 2D, dim=6 in 3D */
  AT_ASSERTM(dets.type().is_cuda(), "dets must be a CUDA tensor");
  AT_ASSERTM(scores.type().is_cuda(), "scores must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(dets.device());

  bool is_3d = dets.size(1) == 6;
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t);

  int dets_num = dets.size(0);

  const int col_blocks = at::cuda::ATenCeilDiv(dets_num, threadsPerBlock);

  at::Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();


  if (is_3d) {
  //std::cout << "performing NMS on 3D boxes in CUDA" << std::endl;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.type(), "nms_kernel_cuda", [&] {
        nms_kernel_3d<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.data_ptr<scalar_t>(),
            (unsigned long long*)mask.data_ptr<int64_t>());
      });
   }
   else {
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.type(), "nms_kernel_cuda", [&] {
        nms_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.data_ptr<scalar_t>(),
            (unsigned long long*)mask.data_ptr<int64_t>());
      });

   }

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
           .to(order_t.device(), keep.scalar_type())});
}