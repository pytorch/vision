#pragma once

// Metal shader source for the roi_align MPS kernels.
//
// Carved out of the shared ops/mps/mps_kernels.h (which is still used by the
// legacy _C ops) so it can be compiled into the stable-ABI _C_stable extension.
// mps_kernels.h opens with #include <ATen/native/mps/OperationUtils.h> and
// instantiates at::native::mps::MetalShaderLibrary, both unavailable under
// -DTORCH_TARGET_VERSION. This header carries only the roi_align Metal source
// as a plain string, handed to aoti_torch_mps_create_shader_library at
// runtime, the same shape PyTorch's AOTInductor MPS backend emits.

namespace vision {
namespace ops {

static const char* roi_align_metal_shader = R"VISION_METAL(

#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

/*----------Helpers--------*/

inline void atomic_add_float(device float* data_ptr, const float val)
{
  atomic_fetch_add_explicit((device atomic_float*) data_ptr, val, memory_order_relaxed);
}


inline void atomic_add_float(device half* data_ptr, const half val)
{
  atomic_fetch_add_explicit((device atomic_float*) data_ptr, static_cast<float>(val), memory_order_relaxed);
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

/*----------Kernels----------*/

template <typename T>
kernel void roi_align(
    constant T       * input          [[buffer(0)]],
    constant T       * rois           [[buffer(1)]],
    device   T       * output         [[buffer(2)]],
    constant float   & spatial_scale  [[buffer(3)]],
    constant int64_t & channels       [[buffer(4)]],
    constant int64_t & height         [[buffer(5)]],
    constant int64_t & width          [[buffer(6)]],
    constant int64_t & pooled_height  [[buffer(7)]],
    constant int64_t & pooled_width   [[buffer(8)]],
    constant int64_t & sampling_ratio [[buffer(9)]],
    constant bool    & aligned        [[buffer(10)]],
    uint     index   [[thread_position_in_grid]])
{
  // Decode linear index into (n, c, ph, pw)
  int64_t pw = index % pooled_width;
  int64_t ph = (index / pooled_width) % pooled_height;
  int64_t c = (index / pooled_width / pooled_height) % channels;
  int64_t n = index / (pooled_width * pooled_height * channels);

  constant T* offset_rois = rois + n * 5;
  int64_t roi_batch_ind = static_cast<int64_t>(offset_rois[0]);

  // Do not using rounding; this implementation detail is critical
  T offset = aligned ? static_cast<T>(0.5) : static_cast<T>(0.0);
  T roi_start_w = offset_rois[1] * spatial_scale - offset;
  T roi_start_h = offset_rois[2] * spatial_scale - offset;
  T roi_end_w   = offset_rois[3] * spatial_scale - offset;
  T roi_end_h   = offset_rois[4] * spatial_scale - offset;

  T roi_width = roi_end_w - roi_start_w;
  T roi_height = roi_end_h - roi_start_h;

  if (!aligned) {
    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, static_cast<T>(1.0));
    roi_height = max(roi_height, static_cast<T>(1.0));
  }

  T bin_size_h = roi_height / static_cast<T>(pooled_height);
  T bin_size_w = roi_width / static_cast<T>(pooled_width);

  constant T* offset_input = input + (roi_batch_ind * channels + c) * height * width;

  // We use roi_bin_grid to sample the grid and mimic integral
  int64_t roi_bin_grid_h = sampling_ratio > 0
    ? sampling_ratio
    : static_cast<int64_t>(ceil(roi_height / static_cast<T>(pooled_height)));
  int64_t roi_bin_grid_w = sampling_ratio > 0
    ? sampling_ratio
    : static_cast<int64_t>(ceil(roi_width / static_cast<T>(pooled_width)));

  // We do average (integral) pooling inside a bin
  // When the grid is empty, output zeros.
  const T count = max(roi_bin_grid_h * roi_bin_grid_w, static_cast<int64_t>(1));
  T output_val = static_cast<T>(0.0);

  for (int64_t iy = 0; iy < roi_bin_grid_h; iy++) {
    T y = roi_start_h + static_cast<T>(ph) * bin_size_h +
          (static_cast<T>(iy) + static_cast<T>(0.5)) * bin_size_h / static_cast<T>(roi_bin_grid_h);
    for (int64_t ix = 0; ix < roi_bin_grid_w; ix++) {
      T x = roi_start_w + static_cast<T>(pw) * bin_size_w +
            (static_cast<T>(ix) + static_cast<T>(0.5)) * bin_size_w / static_cast<T>(roi_bin_grid_w);

      T val = bilinear_interpolate(offset_input, height, width, y, x, index);
      output_val += val;
    }
  }

  output_val /= count;
  output[index] = output_val;
}

#define REGISTER_ROI_ALIGN_OP(DTYPE)       \
template                                              \
[[host_name("roi_align_" #DTYPE)]]                    \
kernel void roi_align<DTYPE>(              \
    constant DTYPE   * input          [[buffer(0)]],  \
    constant DTYPE   * rois           [[buffer(1)]],  \
    device   DTYPE   * output         [[buffer(2)]],  \
    constant float   & spatial_scale  [[buffer(3)]],  \
    constant int64_t & channels       [[buffer(4)]],  \
    constant int64_t & height         [[buffer(5)]],  \
    constant int64_t & width          [[buffer(6)]],  \
    constant int64_t & pooled_height  [[buffer(7)]],  \
    constant int64_t & pooled_width   [[buffer(8)]],  \
    constant int64_t & sampling_ratio [[buffer(9)]],  \
    constant bool    & aligned        [[buffer(10)]], \
    uint     index   [[thread_position_in_grid]]);

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
    uint     index   [[thread_position_in_grid]]){

  // One thread per pooled-output element. Dispatched via dispatchThreads, so
  // each element is processed exactly once; redundant passes would otherwise
  // be silently summed by the atomic_add below (overlapping RoIs share pixels).
  if (index >= static_cast<uint>(output_size)) {
    return;
  }
  {
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
    uint     index   [[thread_position_in_grid]]);

REGISTER_ROI_ALIGN_OP(float);
REGISTER_ROI_ALIGN_OP(half);
REGISTER_ROI_ALIGN_BACKWARD_OP(float, int64_t);
REGISTER_ROI_ALIGN_BACKWARD_OP(half, int64_t);

)VISION_METAL";

} // namespace ops
} // namespace vision
