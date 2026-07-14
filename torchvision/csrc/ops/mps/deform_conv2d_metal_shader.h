#pragma once

// Metal shader source for the deform_conv2d MPS forward kernel (deformable
// im2col). Carved out of ops/mps/mps_kernels.h so it can compile into the
// stable-ABI _C_stable extension without pulling in
// at::native::mps::MetalShaderLibrary (unavailable under
// -DTORCH_TARGET_VERSION).

namespace vision {
namespace ops {

static const char* deform_conv2d_metal_shader = R"VISION_METAL(

#include <metal_stdlib>
using namespace metal;

template <typename T, typename integer_t>
inline T bilinear_interpolate_deformable_conv2d(
    constant T* input,
    integer_t height,
    integer_t width,
    T y,
    T x,
    uint index /* index for debug only*/) {
  if (y <= -1.0 || y >= height || x <= -1.0 || x >= width) {
    return 0;
  }
  integer_t y_low = static_cast<integer_t>(floor(y));
  integer_t x_low = static_cast<integer_t>(floor(x));
  integer_t y_high = y_low + 1;
  integer_t x_high = x_low + 1;

  T ly = y - static_cast<T>(y_low);
  T lx = x - static_cast<T>(x_low);
  T hh = 1.0 - ly;
  T hw = 1.0 - lx;

  T v1 = 0;
  if (y_low >= 0 && x_low >= 0)
    v1 = input[y_low * width + x_low];
  
  T v2 = 0;
  if (y_low >= 0 && x_high <= width - 1)
    v2 = input[y_low * width + x_high];
  
  T v3 = 0;
  if (y_high <= height - 1 && x_low >= 0)
    v3 = input[y_high * width + x_low];
  
  T v4 = 0;
  if (y_high <= height - 1 && x_high <= width - 1)
    v4 = input[y_high * width + x_high];

  T w1 = hh * hw;
  T w2 = hh * lx;
  T w3 = ly * hw;
  T w4 = ly * lx;

  T val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
  return val;
}

template<typename T>
kernel void deformable_im2col_kernel(
    constant T*           input_ptr     [[ buffer(0) ]],
    constant T*           offset_ptr    [[ buffer(1) ]],
    constant T*           mask_ptr      [[ buffer(2) ]],
    constant int2&        input_size    [[ buffer(3) ]],   // (height, width)
    constant int2&        weight_size   [[ buffer(4) ]],   // (weight_h, weight_w)
    constant int2&        pad           [[ buffer(5) ]],   // (pad_h, pad_w)
    constant int2&        stride        [[ buffer(6) ]],   // (stride_h, stride_w)
    constant int2&        dilation      [[ buffer(7) ]],   // (dilation_h, dilation_w)
    constant int&         batch_size    [[ buffer(8) ]],
    constant int&         n_in_channels [[ buffer(9) ]],
    constant int&         n_offset_grps [[ buffer(10)]],
    constant int2&        out_size      [[ buffer(11)]],   // (out_h, out_w)
    constant bool&        use_mask      [[ buffer(12)]],
    device T*             columns_ptr   [[ buffer(13)]],
    uint                  tid           [[ thread_position_in_grid ]],
    uint                  tpg           [[ threads_per_grid ]]
)
{
    int height = input_size.x, width = input_size.y;
    int weight_h = weight_size.x, weight_w = weight_size.y;
    int pad_h = pad.x, pad_w = pad.y;
    int stride_h = stride.x, stride_w = stride.y;
    int dilation_h = dilation.x, dilation_w = dilation.y;
    int out_h = out_size.x, out_w = out_size.y;

    int total = out_w * out_h * batch_size * n_in_channels;
    if (tid >= total) {
        return;
    }

    int out_x = tid % out_w;
    int out_y = (tid / out_w) % out_h;
    int out_b = (tid / (out_w * out_h)) % batch_size;
    int in_c  = tid / (out_w * out_h * batch_size);
    int out_c = in_c * weight_h * weight_w;
    
    int c_per_offset_grp = n_in_channels / n_offset_grps;
    int grp_idx = in_c / c_per_offset_grp;
    
    int col_offset = out_c * (batch_size * out_h * out_w)
                      + out_b * (out_h * out_w)
                      + out_y * out_w + out_x;
    device T* local_columns_ptr = columns_ptr + col_offset;
    
    int input_offset = out_b * (n_in_channels * height * width)
                        + in_c * (height * width);
    constant T* local_input_ptr = input_ptr + input_offset;
    
    int offset_offset = (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;
    constant T* local_offset_ptr = offset_ptr + offset_offset;
    
    constant T* local_mask_ptr = nullptr;
    if (use_mask) {
        int mask_offset = (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;
        local_mask_ptr = mask_ptr + mask_offset;
    }
    
    for (int i = 0; i < weight_h; ++i) {
        for (int j = 0; j < weight_w; ++j) {
            int mask_index = i * weight_w + j;
            int offset_index = 2 * mask_index;
            
            T mask_value = 1;
            if (use_mask) {
                mask_value = local_mask_ptr[mask_index * (out_h * out_w) + out_y * out_w + out_x];
            }
            
            T offset_h_val = local_offset_ptr[offset_index * (out_h * out_w) + out_y * out_w + out_x];
            T offset_w_val = local_offset_ptr[(offset_index + 1) * (out_h * out_w) + out_y * out_w + out_x];
            
            T y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h_val;
            T x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w_val;
            
            T interp = bilinear_interpolate_deformable_conv2d(local_input_ptr, height, width, y, x, tid);
            
            *local_columns_ptr = mask_value * interp;
            
            local_columns_ptr += batch_size * out_h * out_w;
        }
    }
}

#define REGISTER_DEFORMABLE_IM2COL_OP(DTYPE)                                         \
template                                                                             \
[[host_name("deformable_im2col_" #DTYPE)]]                                           \
kernel void deformable_im2col_kernel<DTYPE>(                                         \
    constant DTYPE*               input_ptr        [[ buffer(0) ]],                  \
    constant DTYPE*               offset_ptr       [[ buffer(1) ]],                  \
    constant DTYPE*               mask_ptr         [[ buffer(2) ]],                  \
    constant int2&                input_size       [[ buffer(3) ]],   /* (h, w) */   \
    constant int2&                weight_size      [[ buffer(4) ]],   /* (h, w) */   \
    constant int2&                pad              [[ buffer(5) ]],   /* (h, w) */   \
    constant int2&                stride           [[ buffer(6) ]],   /* (h, w) */   \
    constant int2&                dilation         [[ buffer(7) ]],   /* (h, w) */   \
    constant int&                 batch_size       [[ buffer(8) ]],                  \
    constant int&                 n_in_channels    [[ buffer(9) ]],                  \
    constant int&                 n_offset_grps    [[ buffer(10)]],                  \
    constant int2&                out_size         [[ buffer(11)]],  /* (h, w) */    \
    constant bool&                use_mask         [[ buffer(12)]],                  \
    device DTYPE*                 columns_ptr      [[ buffer(13)]],                  \
    uint                          tid              [[ thread_position_in_grid ]],    \
    uint                          tpg              [[ threads_per_grid ]]);

REGISTER_DEFORMABLE_IM2COL_OP(float);
REGISTER_DEFORMABLE_IM2COL_OP(half);

)VISION_METAL";

} // namespace ops
} // namespace vision
