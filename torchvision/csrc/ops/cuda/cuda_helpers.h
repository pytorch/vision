#pragma once

#include <torch/headeronly/util/Half.h>

namespace vision {
namespace ops {

#define CUDA_1D_KERNEL_LOOP_T(i, n, index_t)                         \
  for (index_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

#define CUDA_1D_KERNEL_LOOP(i, n) CUDA_1D_KERNEL_LOOP_T(i, n, int)

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

// Header-only device atomic add for the backward kernels, modeled on ATen's
// fastAtomicAdd (Half __half2 path + ROCm/pre-sm_70 CAS fallback):
// https://github.com/pytorch/pytorch/blob/b1c216582f0088b17dadda23816b5806f35e5dab/aten/src/ATen/native/cuda/KernelUtils.cuh#L117-L223
// TODO(stable-abi): drop if fastAtomicAdd is promoted into torch/headeronly.
template <typename index_t>
__device__ __forceinline__ void fast_atomic_add(
    float* tensor,
    index_t index,
    const index_t numel,
    float value) {
  atomicAdd(tensor + index, value);
}

template <typename index_t>
__device__ __forceinline__ void fast_atomic_add(
    double* tensor,
    index_t index,
    const index_t numel,
    double value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
  // atomicAdd on double needs sm_60, polyfill with the CAS loop ATen's
  // gpuAtomicAddNoReturn uses below that.
  unsigned long long int* address_as_ull =
      (unsigned long long int*)(tensor + index);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
#else
  atomicAdd(tensor + index, value);
#endif
}

template <typename index_t>
__device__ __forceinline__ void fast_atomic_add(
    torch::headeronly::Half* tensor,
    index_t index,
    const index_t numel,
    torch::headeronly::Half value) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
  // Half atomic add through a 32-bit CAS loop, matching ATen's
  // gpuAtomicAddNoReturn where no fast half atomic exists (ROCm and sm < 70).
  torch::headeronly::Half* address = tensor + index;
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    torch::headeronly::Half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + value;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
#else
  // Pair the half with a zero neighbor to use the fast __half2 atomic, with a
  // scalar-atomic fallback at tensor bounds or odd 16-bit alignment.
  __half* target_addr = reinterpret_cast<__half*>(tensor + index);
  bool low_byte =
      (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && index < (numel - 1)) {
    __half2 value2;
    value2.x = static_cast<__half>(value);
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = static_cast<__half>(value);
    atomicAdd(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
    atomicAdd(
        reinterpret_cast<__half*>(tensor) + index, static_cast<__half>(value));
  }
#endif
}

} // namespace ops
} // namespace vision
