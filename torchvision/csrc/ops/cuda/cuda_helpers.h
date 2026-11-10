#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/macros.h>

#include <cuda_runtime.h>

namespace vision {
namespace ops {

// Returns the current CUDA stream for a device, the stable ABI counterpart of
// at::cuda::getCurrentCUDAStream(). The shim only hands back an opaque pointer
// so the cast happens here once instead of in every kernel launch.
inline cudaStream_t get_current_cuda_stream(int32_t device_index) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

#define CUDA_1D_KERNEL_LOOP_T(i, n, index_t)                         \
  for (index_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

#define CUDA_1D_KERNEL_LOOP(i, n) CUDA_1D_KERNEL_LOOP_T(i, n, int)

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

} // namespace ops
} // namespace vision
