#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline bool devIoU(
    T const* const a,
    T const* const b,
    const float threshold) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
  using acc_T = at::acc_type<T, /*is_cuda=*/true>;
  acc_T interS = (acc_T)width * height;
  acc_T Sa = ((acc_T)a[2] - a[0]) * (a[3] - a[1]);
  acc_T Sb = ((acc_T)b[2] - b[0]) * (b[3] - b[1]);
  return (interS / (Sa + Sb - interS)) > threshold;
}

template <typename T>
__global__ void nms_kernel_impl(
    int n_boxes,
    double iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  const auto row_start = blockIdx.y;
  const auto col_start = blockIdx.x;

  if (row_start > col_start)
    return;

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
    const auto cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU<T>(cur_box, block_boxes + i * 4, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

__global__ static void gather_keep_from_mask(
    bool* keep,
    const unsigned long long* dev_mask,
    const int n_boxes) {
  // Taken and adapted from mmcv
  // https://github.com/open-mmlab/mmcv/blob/03ce9208d18c0a63d7ffa087ea1c2f5661f2441a/mmcv/ops/csrc/common/cuda/nms_cuda_kernel.cuh#L76
  const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
  const auto thread_id = threadIdx.x;

  // Mark the bboxes which have been removed.
  extern __shared__ unsigned long long removed[];

  // Initialize removed.
  for (int i = thread_id; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncthreads();

  for (int nblock = 0; nblock < col_blocks; nblock++) {
    auto removed_val = removed[nblock];
    __syncthreads();
    const int i_offset = nblock * threadsPerBlock;
#pragma unroll
    for (int inblock = 0; inblock < threadsPerBlock; inblock++) {
      const int i = i_offset + inblock;
      if (i >= n_boxes)
        break;
      // Select a candidate, check if it should kept.
      if (!(removed_val & (1ULL << inblock))) {
        if (thread_id == 0) {
          keep[i] = true;
        }
        auto p = dev_mask + i * col_blocks;
        // Remove all bboxes which overlap the candidate.
        for (int j = thread_id; j < col_blocks; j += blockDim.x) {
          if (j >= nblock)
            removed[j] |= p[j];
        }
        __syncthreads();
        removed_val = removed[nblock];
      }
    }
  }
}

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");

  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0))

  at::cuda::CUDAGuard device_guard(dets.device());

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t).contiguous();

  int dets_num = dets.size(0);

  const int col_blocks = ceil_div(dets_num, threadsPerBlock);

  at::Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.scalar_type(), "nms_kernel", [&] {
        nms_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.data_ptr<scalar_t>(),
            (unsigned long long*)mask.data_ptr<int64_t>());
      });

  at::Tensor keep =
      at::zeros({dets_num}, dets.options().dtype(at::kBool).device(at::kCUDA));

  // Unwrap the mask to fill keep with proper values
  // Keeping the unwrap on device instead of applying iterative for loops on cpu
  // prevents the device -> cpu -> device transfer that could be bottleneck for
  // large number of boxes.
  // See https://github.com/pytorch/vision/issues/8713 for more details.
  gather_keep_from_mask<<<
      1,
      min(col_blocks, threadsPerBlock),
      col_blocks * sizeof(unsigned long long),
      stream>>>(
      keep.data_ptr<bool>(),
      (unsigned long long*)mask.data_ptr<int64_t>(),
      dets_num);

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.masked_select(keep);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
}

} // namespace ops
} // namespace vision
