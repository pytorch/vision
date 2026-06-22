#pragma once

// Metal shader source for the nms MPS kernel.
//
// Carved out of the shared ops/mps/mps_kernels.h (which is still used by the
// legacy _C ops) so it can be compiled into the stable-ABI _C_stable extension.
// mps_kernels.h opens with #include <ATen/native/mps/OperationUtils.h> and
// instantiates at::native::mps::MetalShaderLibrary, both unavailable under
// -DTORCH_TARGET_VERSION. This header carries only the nms Metal source as a
// plain string, handed to aoti_torch_mps_create_shader_library at runtime,
// the same shape PyTorch's AOTInductor MPS backend emits.

namespace vision {
namespace ops {

static const char* nms_metal_shader = R"VISION_METAL(

#include <metal_stdlib>
using namespace metal;

/*----------Helpers--------*/

template <typename T>
inline T ceil_div(T n, T m) {
  return (n + m - 1) / m;
}

template <typename T, typename scalar_t>
inline bool IoU(
  constant T & a,
  threadgroup T & b,
  const float threshold) {
  auto xx1 = max(a.x, b.x);
  auto yy1 = max(a.y, b.y);
  auto xx2 = min(a.z, b.z);
  auto yy2 = min(a.w, b.w);
  auto w = max(static_cast<scalar_t>(0), xx2 - xx1);
  auto h = max(static_cast<scalar_t>(0), yy2 - yy1);
  // Upcast to float before multiplications to circumvent precision issues in half.
  auto inter = static_cast<float>(w) * static_cast<float>(h);
  auto area_b = static_cast<float>(b.z - b.x) * static_cast<float>(b.w - b.y);
  auto area_a = static_cast<float>(a.z - a.x) * static_cast<float>(a.w - a.y);
  return (inter / (area_a + area_b - inter)) > threshold;
}

/*----------Kernels----------*/

// This should be in sync with the one in nms_kernel.mm.
// Since metal does not support dynamic array,
// we need to make it static instead of deriving it from [[threads_per_threadgroup]].
constant int64_t nmsThreadsPerBlock = sizeof(uint64_t) * 8;

template<typename T, typename scalar_t>
kernel void nms(constant  T        * dev_boxes     [[buffer(0)]],
                device    uint64_t * mask          [[buffer(1)]],
                constant  int64_t  & n_boxes       [[buffer(2)]],
                constant  float    & iou_threshold [[buffer(3)]],
                uint2     tgid     [[threadgroup_position_in_grid]],
                uint2     tid2     [[thread_position_in_threadgroup]]) {

  const uint row_start = tgid.y;
  const uint col_start = tgid.x;
  const uint tid = tid2.x;
  const uint row_size =
      min(n_boxes - row_start * nmsThreadsPerBlock, nmsThreadsPerBlock);
  const uint col_size =
      min(n_boxes - col_start * nmsThreadsPerBlock, nmsThreadsPerBlock);

  threadgroup T block_boxes[nmsThreadsPerBlock];
  block_boxes[tid] = dev_boxes[nmsThreadsPerBlock * col_start + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < row_size) {
    const uint cur_box_idx = nmsThreadsPerBlock * row_start + tid;
    uint64_t t = 0;
    uint start = 0;

    if (row_start == col_start) {
      start = tid + 1;
    }

    for (uint i = start; i < col_size; i++){
      if (IoU<T, scalar_t>(dev_boxes[cur_box_idx], block_boxes[i], iou_threshold)){
        t |= static_cast<uint64_t>(1) << i;  // discard 1 keep 0
      }
    }
    const uint col_blocks = ceil_div(n_boxes, nmsThreadsPerBlock);
    mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

#define REGISTER_NMS_OP(DTYPE)                             \
template                                                   \
[[host_name("nms_" #DTYPE)]]                               \
kernel void nms<DTYPE ## 4, DTYPE>(                        \
  constant DTYPE ## 4 * dev_boxes         [[buffer(0)]],   \
  device   uint64_t   * mask              [[buffer(1)]],   \
  constant int64_t    & n_boxes           [[buffer(2)]],   \
  constant float      & iou_threshold     [[buffer(3)]],   \
  uint2    tgid   [[threadgroup_position_in_grid]],        \
  uint2    tid2   [[thread_position_in_threadgroup]]);

REGISTER_NMS_OP(float);
REGISTER_NMS_OP(half);

)VISION_METAL";

} // namespace ops
} // namespace vision
