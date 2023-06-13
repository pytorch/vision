#include <ATen/native/mps/OperationUtils.h>

namespace vision {
namespace ops {

namespace mps {

static const char* METAL_VISION = R"VISION_METAL(

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#define MPS_1D_KERNEL_LOOP_T(i, n, n_grids, index_t)                         \
  for (index_t i = (tgid.x * tptg.x) + tid2.x; i < (n); \
       i += (tptg.x * n_grids))

#define MPS_1D_KERNEL_LOOP(i, n, n_grids) MPS_1D_KERNEL_LOOP_T(i, n, n_grids, int)

// This should be in sync with the one in nms_kernel.mm.
// Since metal does not support dynamic array,
// we need to make it static instead of deriving it from [[threads_per_threadgroup]].
constant uint threadsPerBlock = sizeof(uint64_t) * 8;

// Utility functions

template <typename T>
inline T ceil_div(T n, T m) {
  return (n + m - 1) / m;
}

// https://github.com/ShoYamanishi/AppleNumericalComputing/blob/053f06c1f5a831095c4bcc29aaf11366fce5231e/03_dot/metal/dot.metal#L447-L472
void atomic_add_float( device atomic_uint* atom_var, const float val )
{
    uint  fetched_uint,  assigning_uint;
    float fetched_float, assigning_float;

    fetched_uint = atomic_exchange_explicit( atom_var, 0, memory_order_relaxed );

    fetched_float = *( (thread float*) &fetched_uint );

    assigning_float = fetched_float + val;

    assigning_uint =  *( (thread uint*) &assigning_float );

    while ( (fetched_uint = atomic_exchange_explicit( atom_var, assigning_uint, memory_order_relaxed ) ) != 0 )  {

        uint fetched_uint_again = atomic_exchange_explicit( atom_var, 0, memory_order_relaxed );

        float fetched_float_again = *( (thread float*) &fetched_uint_again );

        fetched_float = *( (thread float*) &(fetched_uint) );

        assigning_float = fetched_float_again + fetched_float;

        assigning_uint =  *( (thread uint*) &assigning_float );
    }
}

template <typename T>
inline T bilinear_interpolate(
    constant T* input,
    int64_t height,
    int64_t width,
    T y,
    T x,
    uint index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
void bilinear_interpolate_gradient(
    int height,
    int width,
    T y,
    T x,
    thread T& w1,
    thread T& w2,
    thread T& w3,
    thread T& w4,
    thread int& x_low,
    thread int& x_high,
    thread int& y_low,
    thread int& y_high,
    uint index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
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

template<typename T>
kernel void roi_align(
    constant T       * input         [[buffer(0)]],
    constant T       * rois         [[buffer(1)]],
    device   T       * output        [[buffer(2)]],
    constant int64_t & output_size  [[buffer(3)]],
    constant int64_t & channels  [[buffer(4)]],
    constant int64_t & height  [[buffer(5)]],
    constant int64_t & width  [[buffer(6)]],
    constant int64_t & pooled_height  [[buffer(7)]],
    constant int64_t & pooled_width  [[buffer(8)]],
    constant int64_t & sampling_ratio  [[buffer(9)]],
    constant bool    & aligned  [[buffer(10)]],
    constant float   & spatial_scale  [[buffer(11)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){
  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    constant T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    constant T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros.
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(offset_input, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    output[index] = output_val;
  }
}

#define REGISTER_ROI_ALIGN_OP(DTYPE)                        \
template                                              \
[[host_name("roi_align_" #DTYPE)]]                          \
kernel void roi_align<DTYPE>(                   \
  constant DTYPE * input        [[buffer(0)]],   \
  constant DTYPE * rois         [[buffer(1)]],   \
  device   DTYPE * output        [[buffer(2)]],   \
  constant int64_t & output_size  [[buffer(3)]],   \
  constant int64_t & channels  [[buffer(4)]],   \
  constant int64_t & height  [[buffer(5)]],   \
  constant int64_t & width  [[buffer(6)]],   \
  constant int64_t & pooled_height  [[buffer(7)]],   \
  constant int64_t & pooled_width  [[buffer(8)]],   \
  constant int64_t & sampling_ratio  [[buffer(9)]],   \
  constant bool    & aligned  [[buffer(10)]],   \
  constant float   & spatial_scale  [[buffer(11)]],   \
  uint2     tgid   [[threadgroup_position_in_grid]],  \
  uint2     tptg   [[threads_per_threadgroup]],  \
  uint2     tid2   [[thread_position_in_threadgroup]]);

REGISTER_ROI_ALIGN_OP(float);
REGISTER_ROI_ALIGN_OP(half);

template<typename T>
kernel void roi_align_backward(
    constant T       * grad_output         [[buffer(0)]],
    constant T       * rois         [[buffer(1)]],
    device   T       * grad_input        [[buffer(2)]],
    constant int64_t & output_size  [[buffer(3)]],
    constant int64_t & channels  [[buffer(4)]],
    constant int64_t & height  [[buffer(5)]],
    constant int64_t & width  [[buffer(6)]],
    constant int64_t & pooled_height  [[buffer(7)]],
    constant int64_t & pooled_width  [[buffer(8)]],
    constant int64_t & sampling_ratio  [[buffer(9)]],
    constant bool    & aligned  [[buffer(10)]],
    constant float   & spatial_scale  [[buffer(11)]],
    constant int64_t & n_stride  [[buffer(12)]],
    constant int64_t & c_stride  [[buffer(13)]],
    constant int64_t & h_stride  [[buffer(14)]],
    constant int64_t & w_stride  [[buffer(15)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){

  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    constant T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We need to index the gradient using the tensor strides to access the
    // correct values.
    const int output_offset = n * n_stride + c * c_stride;
    constant T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    const int input_offset = (roi_batch_ind * channels + c) * height * width;

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high,
            index);

        T g1 = grad_output_this_bin * w1 / count;
        T g2 = grad_output_this_bin * w2 / count;
        T g3 = grad_output_this_bin * w3 / count;
        T g4 = grad_output_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          device atomic_uint* xAtomic = (device atomic_uint*)(grad_input + input_offset + y_low * width + x_low);
          device atomic_uint* yAtomic = (device atomic_uint*)(grad_input + input_offset + y_low * width + x_high);
          device atomic_uint* zAtomic = (device atomic_uint*)(grad_input + input_offset + y_high * width + x_low);
          device atomic_uint* wAtomic = (device atomic_uint*)(grad_input + input_offset + y_high * width + x_high);
  
          // atomic_float data type is supported on Metal 3 onward.
          // TODO: Use native atomic_fetch_add_explicit for Metal 3.
          atomic_add_float(xAtomic, static_cast<T>(g1));
          atomic_add_float(yAtomic, static_cast<T>(g2));
          atomic_add_float(zAtomic, static_cast<T>(g3));
          atomic_add_float(wAtomic, static_cast<T>(g4));
          
        } // if
      } // ix
    } // iy
  } // MPS_1D_KERNEL_LOOP
}

#define REGISTER_ROI_ALIGN_BACKWARD_OP(DTYPE) \
template                                        \
[[host_name("roi_align_backward_" #DTYPE)]]    \
kernel void roi_align_backward<DTYPE>(                   \
    constant DTYPE       * grad_output         [[buffer(0)]], \
    constant DTYPE       * rois         [[buffer(1)]], \
    device   DTYPE       * grad_input        [[buffer(2)]], \
    constant int64_t & output_size  [[buffer(3)]], \
    constant int64_t & channels  [[buffer(4)]], \
    constant int64_t & height  [[buffer(5)]], \
    constant int64_t & width  [[buffer(6)]], \
    constant int64_t & pooled_height  [[buffer(7)]], \
    constant int64_t & pooled_width  [[buffer(8)]], \
    constant int64_t & sampling_ratio  [[buffer(9)]], \
    constant bool    & aligned  [[buffer(10)]], \
    constant float   & spatial_scale  [[buffer(11)]],  \
    constant int64_t & n_stride  [[buffer(12)]], \
    constant int64_t & c_stride  [[buffer(13)]], \
    constant int64_t & h_stride  [[buffer(14)]], \
    constant int64_t & w_stride  [[buffer(15)]], \
    uint2     tgid   [[threadgroup_position_in_grid]], \
    uint2     tptg   [[threads_per_threadgroup]], \
    uint2     tid2   [[thread_position_in_threadgroup]]);

REGISTER_ROI_ALIGN_BACKWARD_OP(float);
REGISTER_ROI_ALIGN_BACKWARD_OP(half);

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