#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "cuda_helpers.h"
#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

#include <iostream>
#include <vector>


namespace vision {
namespace ops {

namespace {

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline float devIoU(T const* const a, T const* const b, const float threshold) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
  T interS = width * height;
  T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return interS > threshold * (Sa + Sb - interS);
}

template <typename T>
__global__ void nms_mask_kernel(
    const int num_boxes,
    const float iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  if (row_start > col_start) return;

  const int row_size =
      min(num_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(num_boxes - col_start * threadsPerBlock, threadsPerBlock);

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
      if (devIoU<T>(cur_box, block_boxes + i * 4, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
		// TODO: as a param?
    const int col_blocks = at::cuda::ATenCeilDiv(num_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

typedef uint64_t uint64_type;
typedef uint32_t uint32_type;
struct BitwiseOrArgs {
    uint64_type *dst;
    const uint64_type *src;
    uint32_type size;
};
//! load uint64_type with cache streaming
__device__ __forceinline__ uint64_type load_u64_cs(const uint64_type *ptr) {
    uint64_type val;
    asm volatile("ld.cs.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
    return val;
}


__device__ __forceinline__ void bitwise_or_single_warp(BitwiseOrArgs args) {
    uint64_type * __restrict__ dst = args.dst;
    const uint64_type * __restrict__ src = args.src;
    uint32_type size = args.size;
    for (uint32_type i = threadIdx.x; i < size; i += warpSize) {
        dst[i] |= load_u64_cs(&src[i]);
    }
}

//! true -> ~0, false -> 0
__device__ __forceinline__ uint32_type bool_as_u32_mask(bool v) {
    return (!v) - 1;
}

//! return min value of val in current warp
__device__ __forceinline__ uint32_type warp_reduce_min_brdcst(uint32_type val) {
    __shared__ uint32_type ans;
#pragma unroll
    for (uint32_type offset = warpSize / 2; offset; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset)); // sync?

    if (!threadIdx.x)
        ans = val;
    __syncthreads();
    return ans;
}


__global__ void nms_indices_kernel(
        int nr_boxes,
        const uint64_t * __restrict__ overlap_mask,
				uint64_t *__restrict__ rm_mask,
        int64_t * __restrict__ out_idx) {
    __shared__ uint32_type out_pos;
    __shared__ BitwiseOrArgs bitwise_or_args;

    const uint32_type overlap_mask_width = at::cuda::ATenCeilDiv(nr_boxes, 64),
					nr_box_blocks = at::cuda::ATenCeilDiv(nr_boxes, 64);

    if (!threadIdx.x) {
        uint32_type cnt = nr_box_blocks * 64 - nr_boxes;
        // mark the padded boxes as having been removed
        rm_mask[nr_box_blocks - 1] = ((1ull << cnt) - 1) << (64 - cnt);
        out_pos = 0;
    }
    __syncthreads();

    uint32_type
        box_block_id = threadIdx.x,
        th0_box_block_id = 0;

    while (th0_box_block_id < nr_box_blocks) {
        bool in_range = box_block_id < nr_box_blocks;
        uint64_type cur_mask = ~rm_mask[box_block_id & bool_as_u32_mask(in_range)];
        // remove: 0000; alive: 1111
        //
        // if out of range, to be reduced box_block_id:=11111 (UINT32_MAX)
        // if cur_mask=000000, ...
        uint32_type min_box_block_id = warp_reduce_min_brdcst(
                box_block_id | bool_as_u32_mask(!(in_range && cur_mask)));

        if (min_box_block_id + 1) {
            // min_box_block_id != UINT32_MAX, so at least one thread finds a
            // un-removed box
            if (min_box_block_id == box_block_id) {
                // exactly one thread can take this path
                uint32_type box_id_in_block = __ffsll(cur_mask) - 1,
                         box_id = box_block_id * 64 + box_id_in_block;

                // so this box would not be processed again ??
                rm_mask[box_block_id] |= 1ull << box_id_in_block;

                bitwise_or_args.dst = &rm_mask[box_block_id];
                bitwise_or_args.src =
                    &overlap_mask[box_id * overlap_mask_width + box_block_id];
                bitwise_or_args.size = nr_box_blocks - box_block_id;
                out_idx[out_pos ++] = static_cast<int64_t>(box_id);
            }
            __syncthreads();
            bitwise_or_single_warp(bitwise_or_args);

            // skip the blocks before min_box_block_id
            th0_box_block_id = min_box_block_id;
            box_block_id = min_box_block_id + threadIdx.x;
        } else {
            th0_box_block_id += warpSize;
            box_block_id += warpSize;
        }
    }

    if (!threadIdx.x) {
        out_idx[nr_boxes] = static_cast<int64_t>(out_pos);
    }
}



at::Tensor nms_cuda(const at::Tensor& boxes,
    const at::Tensor& scores,
    float iou_threshold) {
  using scalar_t = float;
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  AT_ASSERTM(scores.type().is_cuda(), "scores must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(boxes.device());

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = at::cuda::ATenCeilDiv(boxes_num, threadsPerBlock);

  at::Tensor mask =
      at::empty({boxes_num * col_blocks}, boxes.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      boxes_sorted.type(), "nms_mask_kernel_cuda", [&] {
        nms_mask_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            boxes_num,
            iou_threshold,
            boxes_sorted.data<scalar_t>(),
            (unsigned long long*)mask.data<int64_t>());
      });

	// mask -- row: boxes_num; col: divup(boxes_num, 64)
	// for box_i<box_j,
	// mask[box_i][box_j // 64]{box_j % 64}: whether box_i suppress box_j
	// (uint64{bit} takes a bit in the uint64)
	// for box_i>box_j, the corresponding mask bit is undefined

	// indices in [0, boxes_num), boxes to keep
	// The last element is the valid length of this vector
  at::Tensor keep = at::empty({boxes_num + 1}, boxes.options().dtype(at::kLong));

	// a bitmask, whether each box is supressed or already processed.
  at::Tensor rm_mask = at::zeros({col_blocks}, boxes.options().dtype(at::kLong));

	nms_indices_kernel<<<1, 32, 0, stream>>>(
			boxes_num,
			(const uint64_t*) mask.data<int64_t>(),
			(uint64_t*) rm_mask.data<int64_t>(),
			keep.data<int64_t>());
	int num_to_keep = *keep[boxes_num].to(at::kCPU).data<int64_t>();
	AT_CUDA_CHECK(cudaGetLastError());
  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)});
}


} // namespace

TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_cuda));
}

} // namespace ops
} // namespace vision
