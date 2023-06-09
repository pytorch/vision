#include <ATen/native/mps/OperationUtils.h>

namespace vision {
namespace ops {

namespace mps {

static const char* METAL_VISION = R"VISION_METAL(

#include <metal_stdlib>
using namespace metal;

constant uint threadsPerBlock = sizeof(uint64_t) * 8;

template <typename T>
inline T ceil_div(T n, T m) {
  return (n + m - 1) / m;
}

template <typename T, typename scalar_t>
bool inline IoU(
  constant T & a,
  threadgroup T & b,
  const float threshold) {
  auto xx1 = max(a.x, b.x);
  auto yy1 = max(a.y, b.y);
  auto xx2 = min(a.z, b.z);
  auto yy2 = min(a.w, b.w);
  auto w = max(static_cast<scalar_t>(0), xx2 - xx1);
  auto h = max(static_cast<scalar_t>(0), yy2 - yy1);
  auto inter = w * h;
  auto area_a = (a.z - a.x) * (a.w - a.y);
  auto area_b = (b.z - b.x) * (b.w - b.y);
  return (inter / (area_a + area_b - inter)) > threshold;
}

template<typename T, typename scalar_t>
kernel void nms(constant  T       * dev_boxes         [[buffer(0)]],
                device    uint64_t * mask           [[buffer(1)]],
                constant  int     & n_boxes      [[buffer(2)]],
                constant  float   & iou_threshold [[buffer(3)]],
                uint2     tgid   [[threadgroup_position_in_grid]],
                uint2     tid2   [[thread_position_in_threadgroup]]) {
  
  const uint row_start = tgid.y;
  const uint col_start = tgid.x;
  const uint tid = tid2.x;
  const uint row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const uint col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  threadgroup T block_boxes[threadsPerBlock];
  block_boxes[tid] = dev_boxes[threadsPerBlock * col_start + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < row_size) {
    const uint cur_box_idx = threadsPerBlock * row_start + tid;
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
    const uint col_blocks = ceil_div(static_cast<uint>(n_boxes), threadsPerBlock);
    mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

#define REGISTER_NMS_OP(DTYPE)                        \
template                                              \
[[host_name("nms_" #DTYPE)]]                          \
kernel void nms<DTYPE ## 4, DTYPE>(                   \
  constant DTYPE ## 4 * dev_boxes        [[buffer(0)]],   \
  device   uint64_t  * mask           [[buffer(1)]],   \
  constant int      & n_boxes        [[buffer(2)]],   \
  constant float    & iou_threshold  [[buffer(3)]],   \
  uint2     tgid   [[threadgroup_position_in_grid]],  \
  uint2     tid2   [[thread_position_in_threadgroup]]);

REGISTER_NMS_OP(float);
REGISTER_NMS_OP(half);

)VISION_METAL";

static id<MTLLibrary> compileBinaryOpsLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> binaryLibrary = nil;
  if (binaryLibrary) {
    return binaryLibrary;
  }

  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  binaryLibrary = [device newLibraryWithSource:[NSString stringWithCString:METAL_VISION encoding:NSASCIIStringEncoding]
                                       options:options
                                         error:&error];
  TORCH_CHECK(binaryLibrary, "Failed to create metal binary library, error: ", [[error description] UTF8String]);
  return binaryLibrary;
}

static id<MTLComputePipelineState> binaryPipelineState(id<MTLDevice> device, const std::string& kernel) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> binaryLib = compileBinaryOpsLibrary(device);
  id<MTLFunction> binaryFunc = [binaryLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(binaryFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:binaryFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
}

}
}
}  // namespace