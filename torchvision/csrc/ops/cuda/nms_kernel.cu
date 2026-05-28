#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Half.h>

#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

namespace stable = torch::stable;
namespace headeronly = torch::headeronly;
using headeronly::ScalarType;
using stable::Tensor;

template <typename T>
struct CudaAccType {
  using type = T;
};

template <>
struct CudaAccType<c10::Half> {
  using type = float;
};

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline bool devIoU(
    T const* const a,
    T const* const b,
    const float threshold) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
  using acc_T = typename CudaAccType<T>::type;
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

inline std::pair<Tensor, Tensor> dispatch_sort(
    const Tensor& self,
    bool is_stable,
    int64_t dim,
    bool descending) {
  constexpr int num_args = 4;
  std::array<StableIValue, num_args> stack{
      stable::detail::from(self),
      stable::detail::from(std::optional<bool>(is_stable)),
      stable::detail::from(dim),
      stable::detail::from(descending)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sort", "stable", stack.data(), TORCH_ABI_VERSION));
  return {
      stable::detail::to<Tensor>(stack[0]),
      stable::detail::to<Tensor>(stack[1])};
}

inline Tensor dispatch_index_select(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  constexpr int num_args = 3;
  std::array<StableIValue, num_args> stack{
      stable::detail::from(self),
      stable::detail::from(dim),
      stable::detail::from(index)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::index_select", "", stack.data(), TORCH_ABI_VERSION));
  return stable::detail::to<Tensor>(stack[0]);
}

inline Tensor dispatch_masked_select(const Tensor& self, const Tensor& mask) {
  constexpr int num_args = 2;
  std::array<StableIValue, num_args> stack{
      stable::detail::from(self), stable::detail::from(mask)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::masked_select", "", stack.data(), TORCH_ABI_VERSION));
  return stable::detail::to<Tensor>(stack[0]);
}

Tensor nms_kernel(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {
  STD_TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
  STD_TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");

  STD_TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  STD_TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  STD_TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  STD_TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0))

  torch::stable::accelerator::DeviceGuard device_guard(dets.get_device_index());

  if (dets.numel() == 0) {
    return stable::new_empty(dets, {0}, ScalarType::Long);
  }

  auto [values, order_t] = dispatch_sort(scores, true, 0, true);
  auto dets_sorted =
      stable::contiguous(dispatch_index_select(dets, 0, order_t));

  int64_t dets_num = dets.size(0);

  const int col_blocks = ceil_div(static_cast<int>(dets_num), threadsPerBlock);

  auto mask =
      stable::new_empty(dets, {dets_num * col_blocks}, ScalarType::Long);

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  void* stream_ptr = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(dets.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  THO_DISPATCH_SWITCH(
      dets_sorted.scalar_type(),
      "nms_kernel",
      THO_DISPATCH_CASE(ScalarType::Float, [&] {
        nms_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.const_data_ptr<scalar_t>(),
            reinterpret_cast<unsigned long long*>(
                mask.mutable_data_ptr<int64_t>()));
      }) THO_DISPATCH_CASE(ScalarType::Double, [&] {
        nms_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.const_data_ptr<scalar_t>(),
            reinterpret_cast<unsigned long long*>(
                mask.mutable_data_ptr<int64_t>()));
      }) THO_DISPATCH_CASE(ScalarType::Half, [&] {
        nms_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.const_data_ptr<scalar_t>(),
            reinterpret_cast<unsigned long long*>(
                mask.mutable_data_ptr<int64_t>()));
      }));

  auto keep = stable::new_zeros(dets, {dets_num}, ScalarType::Bool);

  // Keeping the unwrap on device instead of applying iterative for loops on cpu
  // prevents the device -> cpu -> device transfer that could be bottleneck for
  // large number of boxes.
  // See https://github.com/pytorch/vision/issues/8713 for more details.
  gather_keep_from_mask<<<
      1,
      min(col_blocks, threadsPerBlock),
      col_blocks * sizeof(unsigned long long),
      stream>>>(
      keep.mutable_data_ptr<bool>(),
      reinterpret_cast<unsigned long long*>(mask.mutable_data_ptr<int64_t>()),
      dets_num);

  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return dispatch_masked_select(order_t, keep);
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("nms", TORCH_BOX(&nms_kernel));
}

} // namespace ops
} // namespace vision
