#include "../../StableABICompat.h"
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

using namespace vision::stable;

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline bool devIoU(
    T const* const a,
    T const* const b,
    const float threshold) {
  // Use float for all arithmetic to avoid issues with half operators being disabled
  float a0 = __half2float(a[0]);
  float a1 = __half2float(a[1]);
  float a2 = __half2float(a[2]);
  float a3 = __half2float(a[3]);
  float b0 = __half2float(b[0]);
  float b1 = __half2float(b[1]);
  float b2 = __half2float(b[2]);
  float b3 = __half2float(b[3]);

  float left = max(a0, b0), right = min(a2, b2);
  float top = max(a1, b1), bottom = min(a3, b3);
  float width = max(right - left, 0.0f), height = max(bottom - top, 0.0f);
  float interS = width * height;
  float Sa = (a2 - a0) * (a3 - a1);
  float Sb = (b2 - b0) * (b3 - b1);
  return (interS / (Sa + Sb - interS)) > threshold;
}

// Specialization for float - just use values directly
template <>
__device__ inline bool devIoU<float>(
    float const* const a,
    float const* const b,
    const float threshold) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left, 0.0f), height = max(bottom - top, 0.0f);
  float interS = width * height;
  float Sa = (a[2] - a[0]) * (a[3] - a[1]);
  float Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return (interS / (Sa + Sb - interS)) > threshold;
}

// Specialization for double
template <>
__device__ inline bool devIoU<double>(
    double const* const a,
    double const* const b,
    const float threshold) {
  double left = max(a[0], b[0]), right = min(a[2], b[2]);
  double top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  double width = max(right - left, 0.0), height = max(bottom - top, 0.0);
  double interS = width * height;
  double Sa = (a[2] - a[0]) * (a[3] - a[1]);
  double Sb = (b[2] - b[0]) * (b[3] - b[1]);
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

Tensor nms_kernel(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {
  VISION_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
  VISION_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");

  VISION_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  VISION_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  VISION_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  VISION_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0))

  DeviceGuard device_guard(dets.get_device_index());

  if (dets.numel() == 0) {
    return empty({0}, kLong, Device(kCUDA, dets.get_device_index()));
  }

  // Sort scores descending and get indices
  auto [sorted_scores, order_t] = sort(scores, /*dim=*/0, /*descending=*/true);
  auto dets_sorted = torch::stable::contiguous(index_select(dets, 0, order_t));

  int dets_num = dets.size(0);

  const int col_blocks = ceil_div(dets_num, threadsPerBlock);

  Tensor mask = empty({dets_num * col_blocks}, kLong, Device(kCUDA, dets.get_device_index()));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  // Get CUDA stream
  void* stream_ptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      dets.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  auto dtype = dets_sorted.scalar_type();
  if (dtype == kFloat) {
    nms_kernel_impl<float><<<blocks, threads, 0, stream>>>(
        dets_num,
        iou_threshold,
        dets_sorted.const_data_ptr<float>(),
        (unsigned long long*)mask.mutable_data_ptr<int64_t>());
  } else if (dtype == kDouble) {
    nms_kernel_impl<double><<<blocks, threads, 0, stream>>>(
        dets_num,
        iou_threshold,
        dets_sorted.const_data_ptr<double>(),
        (unsigned long long*)mask.mutable_data_ptr<int64_t>());
  } else {
    VISION_CHECK(false, "nms only supports float and double types");
  }

  Tensor keep = zeros({dets_num}, kBool, Device(kCUDA, dets.get_device_index()));

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
      keep.mutable_data_ptr<bool>(),
      (unsigned long long*)mask.const_data_ptr<int64_t>(),
      dets_num);

  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return masked_select(order_t, keep);
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("nms", TORCH_BOX(&nms_kernel));
}

} // namespace ops
} // namespace vision
