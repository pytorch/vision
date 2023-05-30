#include <ATen/native/mps/OperationUtils.h>

namespace vision {
namespace ops {

namespace mps {

static const char* METAL_VISION = R"VISION_METAL(

#include <metal_stdlib>
using namespace metal;

template <typename T, typename scalar_t>
bool IoU(
  constant T & a,
  constant T & b,
  scalar_t threshold) {
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
kernel void nms(constant  T       * input         [[buffer(0)]],
                device    int64_t * out           [[buffer(1)]],
                constant  int     & dets_num      [[buffer(2)]],
                constant  float   & iou_threshold [[buffer(3)]],
                uint      tid  [[thread_position_in_grid]]) {
  int t = 0;
  for (int i = tid + 1; i < dets_num; i++){
    if (IoU<T, scalar_t>(input[tid], input[i], iou_threshold)){
      t |= static_cast<int>(1) << i;
    }
  }
  out[tid] = static_cast<int64_t>(t);
}

#define REGISTER_NMS_OP(DTYPE)                        \
template                                               \
[[host_name("nms_" #DTYPE)]]                          \
kernel void nms<DTYPE ## 4, DTYPE>(                               \
  constant DTYPE ## 4   * input         [[buffer(0)]],   \
  device   int64_t  * out           [[buffer(1)]],   \
  constant int      & dets_num      [[buffer(2)]],   \
  constant float    & iou_threshold [[buffer(3)]],   \
  uint tid [[thread_position_in_grid]]);

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