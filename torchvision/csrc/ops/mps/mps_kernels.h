#include <ATen/native/mps/OperationUtils.h>

namespace vision {
namespace ops {

namespace mps {

static const char* METAL_VISION = R"VISION_METAL(

#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

/*----------Macros----------*/

#define MPS_1D_KERNEL_LOOP_T(i, n, n_tgs, index_t)      \
  for (index_t i = (tgid.x * tptg.x) + tid2.x; i < (n); \
       i += (tptg.x * n_tgs))

#define MPS_1D_KERNEL_LOOP(i, n, n_tgs) MPS_1D_KERNEL_LOOP_T(i, n, n_tgs, uint)

/*----------Helpers--------*/

template <typename T>
inline T ceil_div(T n, T m) {
  return (n + m - 1) / m;
}

template <typename T>
inline void atomic_add_float( device T* data_ptr, const T val)
{
#if __METAL_VERSION__ >= 300
  // atomic_float is supported in Metal 3 (macOS Ventura) onward.
  device atomic_fetch_add_explicit((device atomic_float*) data_ptr, val, memory_order_relaxed);
#else
  // Custom atomic addition implementation
  // https://github.com/ShoYamanishi/AppleNumericalComputing/blob/053f06c1f5a831095c4bcc29aaf11366fce5231e/03_dot/metal/dot.metal#L447-L472
  // https://forums.developer.nvidia.com/t/atomicadd-float-float-atomicmul-float-float/14639
  // https://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf (See the last slide)
  
  // Create an atomic uint pointer for atomic transaction.
  device atomic_uint* atom_var = (device atomic_uint*)data_ptr;
  // Create necessary storage.
  uint  fetched_uint,  assigning_uint;
  T fetched_float, assigning_float;

  // Replace the value in atom_var with 0 and return the previous value in atom_var.
  fetched_uint = atomic_exchange_explicit( atom_var, 0 /*desired*/, memory_order_relaxed);
  // Read out the previous value as float.
  fetched_float = *( (thread T*) &fetched_uint );

  // Do addition and represent the addition result in uint for atomic transaction.
  assigning_float = fetched_float + val;
  assigning_uint =  *((thread uint*) &assigning_float);

  // atom_var should be 0 now, try to assign the addition result back to the atom_var (data_ptr).
  while ((fetched_uint = atomic_exchange_explicit( atom_var, assigning_uint /*desired*/, memory_order_relaxed)) != 0)  {
    // If atom_var was not 0, i.e. fetched_uint != 0, it means that the data has been modified by other threads.
    // Try to assign 0 and get the previously assigned addition result.
    uint fetched_uint_again = atomic_exchange_explicit(atom_var, 0 /*desired*/, memory_order_relaxed);
    T fetched_float_again = *( (thread T*) &fetched_uint_again );
    // Re-add again
    fetched_float = *((thread T*) &(fetched_uint));
    // Previously assigned addition result + addition result from other threads.
    assigning_float = fetched_float_again + fetched_float;
    assigning_uint =  *( (thread uint*) &assigning_float);
  }
#endif
}

template <typename T, typename integer_t>
inline T bilinear_interpolate(
    constant T* input,
    integer_t height,
    integer_t width,
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

  integer_t y_low = (integer_t)y;
  integer_t x_low = (integer_t)x;
  integer_t y_high;
  integer_t x_high;

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

template <typename T, typename integer_t>
inline void bilinear_interpolate_gradient(
    integer_t height,
    integer_t width,
    T y,
    T x,
    thread T& w1,
    thread T& w2,
    thread T& w3,
    thread T& w4,
    thread integer_t& x_low,
    thread integer_t& x_high,
    thread integer_t& y_low,
    thread integer_t& y_high,
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

  y_low = (integer_t)y;
  x_low = (integer_t)x;

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

template<typename T, typename integer_t>
kernel void roi_align(
    constant T       * input          [[buffer(0)]],
    constant T       * rois           [[buffer(1)]],
    device   T       * output         [[buffer(2)]],
    constant int64_t & output_size    [[buffer(3)]],
    constant int64_t & channels       [[buffer(4)]],
    constant int64_t & height         [[buffer(5)]],
    constant int64_t & width          [[buffer(6)]],
    constant int64_t & pooled_height  [[buffer(7)]],
    constant int64_t & pooled_width   [[buffer(8)]],
    constant int64_t & sampling_ratio [[buffer(9)]],
    constant bool    & aligned        [[buffer(10)]],
    constant float   & spatial_scale  [[buffer(11)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){
  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, c, ph, pw) is an element in the pooled output
    integer_t pw = index % pooled_width;
    integer_t ph = (index / pooled_width) % pooled_height;
    integer_t c = (index / pooled_width / pooled_height) % channels;
    integer_t n = index / pooled_width / pooled_height / channels;

    constant T* offset_rois = rois + n * 5;
    integer_t roi_batch_ind = offset_rois[0];

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
    integer_t roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    integer_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros.
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, static_cast<integer_t>(1)); // e.g. = 4

    T output_val = 0.;
    for (integer_t iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (integer_t ix = 0; ix < roi_bin_grid_w; ix++) {
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

#define REGISTER_ROI_ALIGN_OP(DTYPE, INT_DTYPE)         \
template                                                \
[[host_name("roi_align_" #DTYPE)]]                      \
kernel void roi_align<DTYPE, INT_DTYPE>(                \
  constant DTYPE * input            [[buffer(0)]],      \
  constant DTYPE * rois             [[buffer(1)]],      \
  device   DTYPE * output           [[buffer(2)]],      \
  constant int64_t & output_size    [[buffer(3)]],      \
  constant int64_t & channels       [[buffer(4)]],      \
  constant int64_t & height         [[buffer(5)]],      \
  constant int64_t & width          [[buffer(6)]],      \
  constant int64_t & pooled_height  [[buffer(7)]],      \
  constant int64_t & pooled_width   [[buffer(8)]],      \
  constant int64_t & sampling_ratio [[buffer(9)]],      \
  constant bool    & aligned        [[buffer(10)]],     \
  constant float   & spatial_scale  [[buffer(11)]],     \
  uint2     tgid   [[threadgroup_position_in_grid]],    \
  uint2     tptg   [[threads_per_threadgroup]],         \
  uint2     tid2   [[thread_position_in_threadgroup]]);

template<typename T, typename integer_t>
kernel void roi_align_backward(
    constant T       * grad_output    [[buffer(0)]],
    constant T       * rois           [[buffer(1)]],
    device   T       * grad_input     [[buffer(2)]],
    constant int64_t & output_size    [[buffer(3)]],
    constant int64_t & channels       [[buffer(4)]],
    constant int64_t & height         [[buffer(5)]],
    constant int64_t & width          [[buffer(6)]],
    constant int64_t & pooled_height  [[buffer(7)]],
    constant int64_t & pooled_width   [[buffer(8)]],
    constant int64_t & sampling_ratio [[buffer(9)]],
    constant bool    & aligned        [[buffer(10)]],
    constant float   & spatial_scale  [[buffer(11)]],
    constant int64_t & n_stride       [[buffer(12)]],
    constant int64_t & c_stride       [[buffer(13)]],
    constant int64_t & h_stride       [[buffer(14)]],
    constant int64_t & w_stride       [[buffer(15)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){

  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, c, ph, pw) is an element in the pooled output
    integer_t pw = index % pooled_width;
    integer_t ph = (index / pooled_width) % pooled_height;
    integer_t c = (index / pooled_width / pooled_height) % channels;
    integer_t n = index / pooled_width / pooled_height / channels;

    constant T* offset_rois = rois + n * 5;
    integer_t roi_batch_ind = offset_rois[0];

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
    const integer_t output_offset = n * n_stride + c * c_stride;
    constant T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    integer_t roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    integer_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    const integer_t input_offset = (roi_batch_ind * channels + c) * height * width;

    for (integer_t iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (integer_t ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        integer_t x_low, x_high, y_low, y_high;

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
          atomic_add_float(grad_input + input_offset + y_low * width + x_low, static_cast<T>(g1));
          atomic_add_float(grad_input + input_offset + y_low * width + x_high, static_cast<T>(g2));
          atomic_add_float(grad_input + input_offset + y_high * width + x_low, static_cast<T>(g3));
          atomic_add_float(grad_input + input_offset + y_high * width + x_high, static_cast<T>(g4));
          
        } // if
      } // ix
    } // iy
  } // MPS_1D_KERNEL_LOOP
}

#define REGISTER_ROI_ALIGN_BACKWARD_OP(DTYPE, INT_DTYPE)   \
template                                                   \
[[host_name("roi_align_backward_" #DTYPE)]]                \
kernel void roi_align_backward<DTYPE, INT_DTYPE>(          \
    constant DTYPE   * grad_output    [[buffer(0)]],       \
    constant DTYPE   * rois           [[buffer(1)]],       \
    device   DTYPE   * grad_input     [[buffer(2)]],       \
    constant int64_t & output_size    [[buffer(3)]],       \
    constant int64_t & channels       [[buffer(4)]],       \
    constant int64_t & height         [[buffer(5)]],       \
    constant int64_t & width          [[buffer(6)]],       \
    constant int64_t & pooled_height  [[buffer(7)]],       \
    constant int64_t & pooled_width   [[buffer(8)]],       \
    constant int64_t & sampling_ratio [[buffer(9)]],       \
    constant bool    & aligned        [[buffer(10)]],      \
    constant float   & spatial_scale  [[buffer(11)]],      \
    constant int64_t & n_stride       [[buffer(12)]],      \
    constant int64_t & c_stride       [[buffer(13)]],      \
    constant int64_t & h_stride       [[buffer(14)]],      \
    constant int64_t & w_stride       [[buffer(15)]],      \
    uint2     tgid   [[threadgroup_position_in_grid]],     \
    uint2     tptg   [[threads_per_threadgroup]],          \
    uint2     tid2   [[thread_position_in_threadgroup]]);

template<typename T, typename integer_t>
kernel void roi_pool(
    constant T       * input         [[buffer(0)]],
    constant T       * rois          [[buffer(1)]],
    device   T       * output        [[buffer(2)]],
    device   int64_t * argmax        [[buffer(3)]],
    constant int64_t & output_size   [[buffer(4)]],
    constant int64_t & channels      [[buffer(5)]],
    constant int64_t & height        [[buffer(6)]],
    constant int64_t & width         [[buffer(7)]],
    constant int64_t & pooled_height [[buffer(8)]],
    constant int64_t & pooled_width  [[buffer(9)]],
    constant float   & spatial_scale [[buffer(10)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){
  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, c, ph, pw) is an element in the pooled output
    integer_t pw = index % pooled_width;
    integer_t ph = (index / pooled_width) % pooled_height;
    integer_t c = (index / pooled_width / pooled_height) % channels;
    integer_t n = index / pooled_width / pooled_height / channels;

    constant T* offset_rois = rois + n * 5;
    integer_t roi_batch_ind = offset_rois[0];
    integer_t roi_start_w = round(offset_rois[1] * spatial_scale);
    integer_t roi_start_h = round(offset_rois[2] * spatial_scale);
    integer_t roi_end_w = round(offset_rois[3] * spatial_scale);
    integer_t roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    integer_t roi_width = max(roi_end_w - roi_start_w + 1, static_cast<integer_t>(1));
    integer_t roi_height = max(roi_end_h - roi_start_h + 1, static_cast<integer_t>(1));
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    integer_t hstart = static_cast<integer_t>(floor(static_cast<T>(ph) * bin_size_h));
    integer_t wstart = static_cast<integer_t>(floor(static_cast<T>(pw) * bin_size_w));
    integer_t hend = static_cast<integer_t>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    integer_t wend = static_cast<integer_t>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, static_cast<integer_t>(0)), static_cast<integer_t>(height));
    hend = min(max(hend + roi_start_h, static_cast<integer_t>(0)), static_cast<integer_t>(height));
    wstart = min(max(wstart + roi_start_w, static_cast<integer_t>(0)), static_cast<integer_t>(width));
    wend = min(max(wend + roi_start_w, static_cast<integer_t>(0)), static_cast<integer_t>(width));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    integer_t maxidx = -1;
    constant T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;
    for (integer_t h = hstart; h < hend; ++h) {
      for (integer_t w = wstart; w < wend; ++w) {
        integer_t input_index = h * width + w;
        if (offset_input[input_index] > maxval) {
          maxval = offset_input[input_index];
          maxidx = input_index;
        }
      }
    }
    output[index] = maxval;
    argmax[index] = maxidx;
  }
}

#define REGISTER_ROI_POOL_OP(DTYPE, INT_DTYPE)          \
template                                                \
[[host_name("roi_pool_" #DTYPE)]]                       \
kernel void roi_pool<DTYPE, INT_DTYPE>(                 \
  constant DTYPE * input           [[buffer(0)]],       \
  constant DTYPE * rois            [[buffer(1)]],       \
  device   DTYPE * output          [[buffer(2)]],       \
  device   int64_t * argmax_data   [[buffer(3)]],       \
  constant int64_t & output_size   [[buffer(4)]],       \
  constant int64_t & channels      [[buffer(5)]],       \
  constant int64_t & height        [[buffer(6)]],       \
  constant int64_t & width         [[buffer(7)]],       \
  constant int64_t & pooled_height [[buffer(8)]],       \
  constant int64_t & pooled_width  [[buffer(9)]],       \
  constant float   & spatial_scale [[buffer(10)]],      \
  uint2     tgid   [[threadgroup_position_in_grid]],    \
  uint2     tptg   [[threads_per_threadgroup]],         \
  uint2     tid2   [[thread_position_in_threadgroup]]);

template<typename T, typename integer_t>
kernel void roi_pool_backward(
    constant T       * grad_output   [[buffer(0)]],
    constant T       * rois          [[buffer(1)]],
    constant int64_t * argmax_data   [[buffer(2)]],
    device   T       * grad_input    [[buffer(3)]],
    constant int64_t & output_size   [[buffer(4)]],
    constant int64_t & channels      [[buffer(5)]],
    constant int64_t & height        [[buffer(6)]],
    constant int64_t & width         [[buffer(7)]],
    constant int64_t & pooled_height [[buffer(8)]],
    constant int64_t & pooled_width  [[buffer(9)]],
    constant float   & spatial_scale [[buffer(10)]],
    constant int64_t & n_stride      [[buffer(11)]],
    constant int64_t & c_stride      [[buffer(12)]],
    constant int64_t & h_stride      [[buffer(13)]],
    constant int64_t & w_stride      [[buffer(14)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){

  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, c, ph, pw) is an element in the pooled output
    integer_t pw = index % pooled_width;
    integer_t ph = (index / pooled_width) % pooled_height;
    integer_t c = (index / pooled_width / pooled_height) % channels;
    integer_t n = index / pooled_width / pooled_height / channels;

    constant T* offset_rois = rois + n * 5;
    integer_t roi_batch_ind = offset_rois[0];

    const integer_t output_offset = n * n_stride + c * c_stride;
    constant integer_t * argmax_data_offset =
        argmax_data + (n * channels + c) * pooled_height * pooled_width;
    const integer_t argmax = argmax_data_offset[ph * pooled_width + pw];
    const integer_t offset = (roi_batch_ind * channels + c) * height * width;

    if (argmax != -1) {
      atomic_add_float(grad_input + offset + argmax, static_cast<T>(grad_output[output_offset + ph * h_stride + pw * w_stride]));
    }
    
  } // MPS_1D_KERNEL_LOOP
}

#define REGISTER_ROI_POOL_BACKWARD_OP(DTYPE, INT_DTYPE)   \
template                                                  \
[[host_name("roi_pool_backward_" #DTYPE)]]                \
kernel void roi_pool_backward<DTYPE, INT_DTYPE>(          \
    constant DTYPE   * grad_output   [[buffer(0)]],       \
    constant DTYPE   * rois          [[buffer(1)]],       \
    constant int64_t * argmax_data   [[buffer(2)]],       \
    device   DTYPE   * grad_input    [[buffer(3)]],       \
    constant int64_t & output_size   [[buffer(4)]],       \
    constant int64_t & channels      [[buffer(5)]],       \
    constant int64_t & height        [[buffer(6)]],       \
    constant int64_t & width         [[buffer(7)]],       \
    constant int64_t & pooled_height [[buffer(8)]],       \
    constant int64_t & pooled_width  [[buffer(9)]],       \
    constant float   & spatial_scale [[buffer(10)]],      \
    constant int64_t & n_stride      [[buffer(11)]],      \
    constant int64_t & c_stride      [[buffer(12)]],      \
    constant int64_t & h_stride      [[buffer(13)]],      \
    constant int64_t & w_stride      [[buffer(14)]],      \
    uint2     tgid   [[threadgroup_position_in_grid]],    \
    uint2     tptg   [[threads_per_threadgroup]],         \
    uint2     tid2   [[thread_position_in_threadgroup]]);

template<typename T, typename integer_t>
kernel void ps_roi_align(
    constant T       * input           [[buffer(0)]],
    constant T       * rois            [[buffer(1)]],
    device   T       * output          [[buffer(2)]],
    device   int64_t * channel_mapping [[buffer(3)]],
    constant int64_t & output_size     [[buffer(4)]],
    constant int64_t & channels        [[buffer(5)]],
    constant int64_t & height          [[buffer(6)]],
    constant int64_t & width           [[buffer(7)]],
    constant int64_t & pooled_height   [[buffer(8)]],
    constant int64_t & pooled_width    [[buffer(9)]],
    constant int64_t & sampling_ratio  [[buffer(10)]],
    constant int64_t & channels_out    [[buffer(11)]],
    constant float   & spatial_scale   [[buffer(12)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){
  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, c_out, ph, pw) is an element in the pooled output
    integer_t pw = index % pooled_width;
    integer_t ph = (index / pooled_width) % pooled_height;
    integer_t c_out = (index / pooled_width / pooled_height) % channels_out;
    integer_t n = index / pooled_width / pooled_height / channels_out;

    // (n, c_in, ph, pw) is the associated element in the input
    integer_t c_in = (c_out * pooled_height + ph) * pooled_width + pw;

    // [start, end) interval for spatial sampling
    constant T* offset_rois = rois + n * 5;
    integer_t roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_rois[1] * spatial_scale - static_cast<T>(0.5);
    T roi_start_h = offset_rois[2] * spatial_scale - static_cast<T>(0.5);
    T roi_end_w = offset_rois[3] * spatial_scale - static_cast<T>(0.5);
    T roi_end_h = offset_rois[4] * spatial_scale - static_cast<T>(0.5);

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // Do not using floor/ceil; this implementation detail is critical
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;

    // We use roi_bin_grid to sample the grid and mimic integral
    integer_t roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);
    integer_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T count = roi_bin_grid_h * roi_bin_grid_w;

    constant T* offset_input =
        input + (roi_batch_ind * channels + c_in) * height * width;
    T out_sum = 0;
    for (integer_t iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = hstart +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);
      for (integer_t ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = wstart +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);
        T val = bilinear_interpolate(offset_input, height, width, y, x, index);
        out_sum += val;
      }
    }

    out_sum /= count;
    output[index] = out_sum;
    channel_mapping[index] = c_in;
  }
}

#define REGISTER_PS_ROI_ALIGN_OP(DTYPE, INT_DTYPE)      \
template                                                \
[[host_name("ps_roi_align_" #DTYPE)]]                   \
kernel void ps_roi_align<DTYPE, INT_DTYPE>(             \
  constant DTYPE   * input           [[buffer(0)]],     \
  constant DTYPE   * rois            [[buffer(1)]],     \
  device   DTYPE   * output          [[buffer(2)]],     \
  device   int64_t * channel_mapping [[buffer(3)]],     \
  constant int64_t & output_size     [[buffer(4)]],     \
  constant int64_t & channels        [[buffer(5)]],     \
  constant int64_t & height          [[buffer(6)]],     \
  constant int64_t & width           [[buffer(7)]],     \
  constant int64_t & pooled_height   [[buffer(8)]],     \
  constant int64_t & pooled_width    [[buffer(9)]],     \
  constant int64_t & sampling_ratio  [[buffer(10)]],    \
  constant int64_t & channels_out    [[buffer(11)]],    \
  constant float   & spatial_scale   [[buffer(12)]],    \
  uint2     tgid   [[threadgroup_position_in_grid]],    \
  uint2     tptg   [[threads_per_threadgroup]],         \
  uint2     tid2   [[thread_position_in_threadgroup]]);

template<typename T, typename integer_t>
kernel void ps_roi_align_backward(
    constant T       * grad_output     [[buffer(0)]],
    constant T       * rois            [[buffer(1)]],
    constant int64_t * channel_mapping [[buffer(2)]],
    device   T       * grad_input      [[buffer(3)]],
    constant int64_t & output_size     [[buffer(4)]],
    constant int64_t & channels        [[buffer(5)]],
    constant int64_t & height          [[buffer(6)]],
    constant int64_t & width           [[buffer(7)]],
    constant int64_t & pooled_height   [[buffer(8)]],
    constant int64_t & pooled_width    [[buffer(9)]],
    constant int64_t & sampling_ratio  [[buffer(10)]],
    constant int64_t & channels_out    [[buffer(11)]],
    constant float   & spatial_scale   [[buffer(12)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){

  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, *, ph, pw) is an element in the pooled output
    integer_t pw = index % pooled_width;
    integer_t ph = (index / pooled_width) % pooled_height;
    integer_t n = index / pooled_width / pooled_height / channels_out;

    constant T* offset_rois = rois + n * 5;
    integer_t roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_rois[1] * spatial_scale - static_cast<T>(0.5);
    T roi_start_h = offset_rois[2] * spatial_scale - static_cast<T>(0.5);
    T roi_end_w = offset_rois[3] * spatial_scale - static_cast<T>(0.5);
    T roi_end_h = offset_rois[4] * spatial_scale - static_cast<T>(0.5);

    // Force too small ROIs to be 1x1
    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    integer_t c_in = channel_mapping[index];

    // Do not using floor/ceil; this implementation detail is critical
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;

    const T grad_output_this_bin = grad_output[index];

    // We use roi_bin_grid to sample the grid and mimic integral
    integer_t roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    integer_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T count = roi_bin_grid_h * roi_bin_grid_w;

    const integer_t offset = (roi_batch_ind * channels + c_in) * height * width;

    for (integer_t iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = hstart +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);
      for (integer_t ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = wstart +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        integer_t x_low, x_high, y_low, y_high;

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
          atomic_add_float(grad_input + offset + y_low * width + x_low, static_cast<T>(g1));
          atomic_add_float(grad_input + offset + y_low * width + x_high, static_cast<T>(g2));
          atomic_add_float(grad_input + offset + y_high * width + x_low, static_cast<T>(g3));
          atomic_add_float(grad_input + offset + y_high * width + x_high, static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  }
}

#define REGISTER_PS_ROI_ALIGN_BACKWARD_OP(DTYPE, INT_DTYPE)   \
template                                                      \
[[host_name("ps_roi_align_backward_" #DTYPE)]]                \
kernel void ps_roi_align_backward<DTYPE, INT_DTYPE>(          \
    constant DTYPE   * grad_output     [[buffer(0)]],         \
    constant DTYPE   * rois            [[buffer(1)]],         \
    constant int64_t * channel_mapping [[buffer(2)]],         \
    device   DTYPE   * grad_input      [[buffer(3)]],         \
    constant int64_t & output_size     [[buffer(4)]],         \
    constant int64_t & channels        [[buffer(5)]],         \
    constant int64_t & height          [[buffer(6)]],         \
    constant int64_t & width           [[buffer(7)]],         \
    constant int64_t & pooled_height   [[buffer(8)]],         \
    constant int64_t & pooled_width    [[buffer(9)]],         \
    constant int64_t & sampling_ratio  [[buffer(10)]],        \
    constant int64_t & channels_out    [[buffer(11)]],        \
    constant float   & spatial_scale   [[buffer(12)]],        \
    uint2     tgid   [[threadgroup_position_in_grid]],        \
    uint2     tptg   [[threads_per_threadgroup]],             \
    uint2     tid2   [[thread_position_in_threadgroup]]);

template<typename T, typename integer_t>
kernel void ps_roi_pool(
    constant T       * input           [[buffer(0)]],
    constant T       * rois            [[buffer(1)]],
    device   T       * output          [[buffer(2)]],
    device   int64_t * channel_mapping [[buffer(3)]],
    constant int64_t & output_size     [[buffer(4)]],
    constant int64_t & channels        [[buffer(5)]],
    constant int64_t & height          [[buffer(6)]],
    constant int64_t & width           [[buffer(7)]],
    constant int64_t & pooled_height   [[buffer(8)]],
    constant int64_t & pooled_width    [[buffer(9)]],
    constant int64_t & channels_out    [[buffer(10)]],
    constant float   & spatial_scale   [[buffer(11)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){
  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, c_out, ph, pw) is an element in the pooled output
    integer_t pw = index % pooled_width;
    integer_t ph = (index / pooled_width) % pooled_height;
    integer_t c_out = (index / (pooled_width * pooled_height)) % channels_out;
    integer_t n = index / pooled_width / pooled_height / channels_out;

    // (n, c_in, ph, pw) is the associated element in the input
    integer_t c_in = (c_out * pooled_height + ph) * pooled_width + pw;

    // [start, end) interval for spatial sampling
    constant T* offset_rois = rois + n * 5;
    integer_t roi_batch_ind = offset_rois[0];
    integer_t roi_start_w = round(offset_rois[1] * spatial_scale);
    integer_t roi_start_h = round(offset_rois[2] * spatial_scale);
    integer_t roi_end_w = round(offset_rois[3] * spatial_scale);
    integer_t roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    integer_t roi_width = max(roi_end_w - roi_start_w, static_cast<integer_t>(1));
    integer_t roi_height = max(roi_end_h - roi_start_h, static_cast<integer_t>(1));
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    integer_t hstart = static_cast<integer_t>(floor(static_cast<T>(ph) * bin_size_h));
    integer_t wstart = static_cast<integer_t>(floor(static_cast<T>(pw) * bin_size_w));
    integer_t hend = static_cast<integer_t>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    integer_t wend = static_cast<integer_t>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, static_cast<integer_t>(0)), static_cast<integer_t>(height - 1));
    hend = min(max(hend + roi_start_h, static_cast<integer_t>(0)), static_cast<integer_t>(height - 1));
    wstart = min(max(wstart + roi_start_w, static_cast<integer_t>(0)), static_cast<integer_t>(width - 1));
    wend = min(max(wend + roi_start_w, static_cast<integer_t>(0)), static_cast<integer_t>(width - 1));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    constant T* offset_input =
        input + (roi_batch_ind * channels + c_in) * height * width;
    T out_sum = 0;
    for (integer_t h = hstart; h < hend; ++h) {
      for (integer_t w = wstart; w < wend; ++w) {
        integer_t input_index = h * width + w;
        out_sum += offset_input[input_index];
      }
    }

    T bin_area = (hend - hstart) * (wend - wstart);
    output[index] = is_empty ? static_cast<T>(0) : out_sum / bin_area;
    channel_mapping[index] = c_in;
  }
}

#define REGISTER_PS_ROI_POOL_OP(DTYPE, INT_DTYPE)     \
template                                              \
[[host_name("ps_roi_pool_" #DTYPE)]]                  \
kernel void ps_roi_pool<DTYPE, INT_DTYPE>(            \
  constant DTYPE   * input           [[buffer(0)]],   \
  constant DTYPE   * rois            [[buffer(1)]],   \
  device   DTYPE   * output          [[buffer(2)]],   \
  device   int64_t * channel_mapping [[buffer(3)]],   \
  constant int64_t & output_size     [[buffer(4)]],   \
  constant int64_t & channels        [[buffer(5)]],   \
  constant int64_t & height          [[buffer(6)]],   \
  constant int64_t & width           [[buffer(7)]],   \
  constant int64_t & pooled_height   [[buffer(8)]],   \
  constant int64_t & pooled_width    [[buffer(9)]],   \
  constant int64_t & channels_out    [[buffer(10)]],  \
  constant float   & spatial_scale   [[buffer(11)]],  \
  uint2    tgid   [[threadgroup_position_in_grid]],   \
  uint2    tptg   [[threads_per_threadgroup]],        \
  uint2    tid2   [[thread_position_in_threadgroup]]);

template<typename T, typename integer_t>
kernel void ps_roi_pool_backward(
    constant T       * grad_output     [[buffer(0)]],
    constant T       * rois            [[buffer(1)]],
    constant int64_t * channel_mapping [[buffer(2)]],
    device   T       * grad_input      [[buffer(3)]],
    constant int64_t & output_size     [[buffer(4)]],
    constant int64_t & channels        [[buffer(5)]],
    constant int64_t & height          [[buffer(6)]],
    constant int64_t & width           [[buffer(7)]],
    constant int64_t & pooled_height   [[buffer(8)]],
    constant int64_t & pooled_width    [[buffer(9)]],
    constant int64_t & channels_out    [[buffer(10)]],
    constant float   & spatial_scale   [[buffer(11)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){

  MPS_1D_KERNEL_LOOP(index, output_size, 1) {
    // (n, *, ph, pw) is an element in the pooled output
    integer_t pw = index % pooled_width;
    integer_t ph = (index / pooled_width) % pooled_height;
    integer_t n = index / pooled_width / pooled_height / channels_out;

    constant T* offset_rois = rois + n * 5;
    integer_t roi_batch_ind = offset_rois[0];
    integer_t roi_start_w = round(offset_rois[1] * spatial_scale);
    integer_t roi_start_h = round(offset_rois[2] * spatial_scale);
    integer_t roi_end_w = round(offset_rois[3] * spatial_scale);
    integer_t roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    integer_t roi_width = max(roi_end_w - roi_start_w, static_cast<integer_t>(1));
    integer_t roi_height = max(roi_end_h - roi_start_h, static_cast<integer_t>(1));
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    integer_t hstart = static_cast<integer_t>(floor(static_cast<T>(ph) * bin_size_h));
    integer_t wstart = static_cast<integer_t>(floor(static_cast<T>(pw) * bin_size_w));
    integer_t hend = static_cast<integer_t>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    integer_t wend = static_cast<integer_t>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, static_cast<integer_t>(0)), static_cast<integer_t>(height));
    hend = min(max(hend + roi_start_h, static_cast<integer_t>(0)), static_cast<integer_t>(height));
    wstart = min(max(wstart + roi_start_w, static_cast<integer_t>(0)), static_cast<integer_t>(width));
    wend = min(max(wend + roi_start_w, static_cast<integer_t>(0)), static_cast<integer_t>(width));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    integer_t c_in = channel_mapping[index];
    T bin_area = (hend - hstart) * (wend - wstart);
    T diff_val = is_empty ? static_cast<T>(0) : grad_output[index] / bin_area;

    const integer_t offset = (roi_batch_ind * channels + c_in) * height * width;

    for (integer_t h = hstart; h < hend; ++h) {
      for (integer_t w = wstart; w < wend; ++w) {
        integer_t grad_input_index = h * width + w;
        atomic_add_float(grad_input + offset + grad_input_index, diff_val);
      }
    }
    
  } // MPS_1D_KERNEL_LOOP
}

#define REGISTER_PS_ROI_POOL_BACKWARD_OP(DTYPE, INT_DTYPE)   \
template                                                     \
[[host_name("ps_roi_pool_backward_" #DTYPE)]]                \
kernel void ps_roi_pool_backward<DTYPE, INT_DTYPE>(          \
    constant DTYPE   * grad_output     [[buffer(0)]],        \
    constant DTYPE   * rois            [[buffer(1)]],        \
    constant int64_t * channel_mapping [[buffer(2)]],        \
    device   DTYPE   * grad_input      [[buffer(3)]],        \
    constant int64_t & output_size     [[buffer(4)]],        \
    constant int64_t & channels        [[buffer(5)]],        \
    constant int64_t & height          [[buffer(6)]],        \
    constant int64_t & width           [[buffer(7)]],        \
    constant int64_t & pooled_height   [[buffer(8)]],        \
    constant int64_t & pooled_width    [[buffer(9)]],        \
    constant int64_t & channels_out    [[buffer(10)]],       \ 
    constant float   & spatial_scale   [[buffer(11)]],       \
    uint2     tgid   [[threadgroup_position_in_grid]],       \
    uint2     tptg   [[threads_per_threadgroup]],            \
    uint2     tid2   [[thread_position_in_threadgroup]]);

REGISTER_NMS_OP(float);
REGISTER_NMS_OP(half);
REGISTER_ROI_ALIGN_OP(float, int64_t);
REGISTER_ROI_ALIGN_OP(half, int64_t);
REGISTER_ROI_ALIGN_BACKWARD_OP(float, int64_t);
REGISTER_ROI_ALIGN_BACKWARD_OP(half, int64_t);
REGISTER_ROI_POOL_OP(float, int64_t);
REGISTER_ROI_POOL_OP(half, int64_t);
REGISTER_ROI_POOL_BACKWARD_OP(float, int64_t);
REGISTER_ROI_POOL_BACKWARD_OP(half, int64_t);
REGISTER_PS_ROI_ALIGN_OP(float, int64_t);
REGISTER_PS_ROI_ALIGN_OP(half, int64_t);
REGISTER_PS_ROI_ALIGN_BACKWARD_OP(float, int64_t);
REGISTER_PS_ROI_ALIGN_BACKWARD_OP(half, int64_t);
REGISTER_PS_ROI_POOL_OP(float, int64_t);
REGISTER_PS_ROI_POOL_OP(half, int64_t);
REGISTER_PS_ROI_POOL_BACKWARD_OP(float, int64_t);
REGISTER_PS_ROI_POOL_BACKWARD_OP(half, int64_t);

)VISION_METAL";

static id<MTLLibrary> compileVisionOpsLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> visionLibrary = nil;
  if (visionLibrary) {
    return visionLibrary;
  }

  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  visionLibrary = [device newLibraryWithSource:[NSString stringWithCString:METAL_VISION encoding:NSASCIIStringEncoding]
                                       options:options
                                         error:&error];
  TORCH_CHECK(visionLibrary, "Failed to create metal vision library, error: ", [[error description] UTF8String]);
  return visionLibrary;
}

static id<MTLComputePipelineState> visionPipelineState(id<MTLDevice> device, const std::string& kernel) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> visionLib = compileVisionOpsLibrary(device);
  id<MTLFunction> visionFunc = [visionLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(visionFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:visionFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
}

} // namespace mps
} // namespace ops
} // namespace vision
