#pragma once

// Metal shader source for the ps_roi_pool MPS kernels.
//
// Carved out of the shared ops/mps/mps_kernels.h (which is still used by the
// legacy _C ops) so it can be compiled into the stable-ABI _C_stable extension.
// mps_kernels.h opens with #include <ATen/native/mps/OperationUtils.h> and
// instantiates at::native::mps::MetalShaderLibrary, both unavailable under
// -DTORCH_TARGET_VERSION. This header carries only the ps_roi_pool Metal
// source as a plain string, handed to aoti_torch_mps_create_shader_library at
// runtime, the same shape PyTorch's AOTInductor MPS backend emits.

namespace vision {
namespace ops {

static const char* ps_roi_pool_metal_shader = R"VISION_METAL(
#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

/*----------Macros----------*/

#define MPS_1D_KERNEL_LOOP_T(i, n, n_tgs, index_t)      \
  for (index_t i = (tgid.x * tptg.x) + tid2.x; i < (n); \
       i += (tptg.x * n_tgs))

#define MPS_1D_KERNEL_LOOP(i, n, n_tgs) MPS_1D_KERNEL_LOOP_T(i, n, n_tgs, uint)

/*----------Helpers--------*/

inline void atomic_add_float(device float* data_ptr, const float val)
{
  atomic_fetch_add_explicit((device atomic_float*) data_ptr, val, memory_order_relaxed);
}


inline void atomic_add_float(device half* data_ptr, const half val)
{
  atomic_fetch_add_explicit((device atomic_float*) data_ptr, static_cast<float>(val), memory_order_relaxed);
}

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
    uint     index   [[thread_position_in_grid]]){

  // One thread per pooled-output element. Dispatched via dispatchThreads, so
  // each element is processed exactly once; redundant passes would otherwise
  // be silently summed by the atomic_add below (overlapping RoIs share pixels).
  if (index >= static_cast<uint>(output_size)) {
    return;
  }
  {
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
    uint     index   [[thread_position_in_grid]]);

REGISTER_PS_ROI_POOL_OP(float, int64_t);
REGISTER_PS_ROI_POOL_OP(half, int64_t);
REGISTER_PS_ROI_POOL_BACKWARD_OP(float, int64_t);
REGISTER_PS_ROI_POOL_BACKWARD_OP(half, int64_t);
)VISION_METAL";

} // namespace ops
} // namespace vision
