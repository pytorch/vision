//
//  Deform_conv2d_kernals.metal
//  torchvision
//
//  Created by Thomas Martin on 14/10/2024.
//

// This include will only work when the remaining code is embedded in a C string in mps_kernels.h
//#include <ATen/native/mps/OperationUtils.h>

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

// **********************************************************************
// MACROS AND HELPER FUNCTIONS SHOULD NOT BE INCLUDED IN THE FINAL SOURCE
// AS THEY ARE ALREADY INCLUDED IN mps_kernels.h
// **********************************************************************

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
     atomic_fetch_add_explicit((device atomic_float*) data_ptr, val, memory_order_relaxed);
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


template <typename scalar_t, typename index_t>
kernel void deformable_im2col(
    index_t n                       [[buffer(0)]],
    constant scalar_t* input_ptr    [[buffer(1)]],
    constant scalar_t* offset_ptr   [[buffer(2)]],
    constant scalar_t* mask_ptr     [[buffer(3)]],
    index_t height                  [[buffer(4)]],
    index_t width                   [[buffer(5)]],
    index_t weight_h                [[buffer(6)]],
    index_t weight_w                [[buffer(7)]],
    index_t pad_h                   [[buffer(8)]],
    index_t pad_w                   [[buffer(9)]],
    index_t stride_h                [[buffer(10)]],
    index_t stride_w                [[buffer(11)]],
    index_t dilation_h              [[buffer(12)]],
    index_t dilation_w              [[buffer(13)]],
    index_t batch_sz                [[buffer(14)]], // parallel_imgs
    index_t n_in_channels           [[buffer(15)]],
    index_t n_offset_grps           [[buffer(16)]], //deformable_grp
    index_t out_h                   [[buffer(17)]],
    index_t out_w                   [[buffer(18)]],
    constant bool & use_mask        [[buffer(19)]],
    device scalar_t* columns_ptr    [[buffer(20)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]) {
        MPS_1D_KERNEL_LOOP(index, n, 1) {
            const index_t out_x = index % out_w;
            const index_t out_y = (index / out_w) % out_h;
            const index_t out_b = (index / (out_w * out_h)) % batch_sz;
            const index_t in_c = index / (out_w * out_h * batch_sz);
            const index_t out_c = in_c * weight_h * weight_w;
            
            index_t c_per_offset_grp = n_in_channels / n_offset_grps;
            const index_t grp_idx = in_c / c_per_offset_grp;
            
            columns_ptr +=
            (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
             out_y * out_w + out_x);
            
            input_ptr +=
            (out_b * (n_in_channels * height * width) + in_c * (height * width));
            
            offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
            out_h * out_w;
            
            if (use_mask) {
                mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
                out_h * out_w;
            }
            
            for (int i = 0; i < weight_h; ++i) {
                for (int j = 0; j < weight_w; ++j) {
                    const index_t mask_idx = i * weight_w + j;
                    const index_t offset_idx = 2 * mask_idx;
                    
                    scalar_t mask_value = 1;
                    if (use_mask) {
                        mask_value =
                        mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
                    }
                    
                    const scalar_t offset_h =
                    offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
                    const scalar_t offset_w = offset_ptr
                    [(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
                    const scalar_t y =
                    (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
                    const scalar_t x =
                    (out_x * stride_w - pad_w) + j * dilation_w + offset_w;
                    *columns_ptr =
                    mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
                    columns_ptr += batch_sz * out_h * out_w;
                }
            }
        }
        
}

#define REGISTER_DEFORMABLE_IM2COL_OP(DTYPE)   \
template                                  \
[[host_name("deformable_im2col_" #DTYPE)]] \
template <typename scalar_t, typename index_t> \
kernel void deformable_im2col(   \
index_t n                       [[buffer(0)]], \
constant scalar_t* input_ptr    [[buffer(1)]], \
constant scalar_t* offset_ptr   [[buffer(2)]],  \
constant scalar_t* mask_ptr     [[buffer(3)]],  \
index_t height                  [[buffer(4)]],  \
index_t width                   [[buffer(5)]],  \
index_t weight_h                [[buffer(6)]],  \
index_t weight_w                [[buffer(7)]],  \
index_t pad_h                   [[buffer(8)]],  \
index_t pad_w                   [[buffer(9)]],  \
index_t stride_h                [[buffer(10)]], \
index_t stride_w                [[buffer(11)]], \
index_t dilation_h              [[buffer(12)]], \
index_t dilation_w              [[buffer(13)]], \
index_t batch_sz                [[buffer(14)]], \
index_t n_in_channels           [[buffer(15)]], \
index_t n_offset_grps           [[buffer(16)]], \
index_t out_h                   [[buffer(17)]], \
index_t out_w                   [[buffer(18)]], \
constant bool & use_mask        [[buffer(19)]], \
device scalar_t* columns_ptr    [[buffer(20)]], \
uint2     tgid   [[threadgroup_position_in_grid]],  \
uint2     tptg   [[threads_per_threadgroup]],       \
uint2     tid2   [[thread_position_in_threadgroup]]);
                                         

        
        
        
        


template <typename scalar_t, typename index_t>
kernel void deformable_col2im(
    index_t n                               [[buffer(0)]],
    constant scalar_t* col                  [[buffer(1)]],
    constant scalar_t* offset_ptr           [[buffer(2)]],
    constant scalar_t* mask_ptr             [[buffer(3)]],
    index_t channels                        [[buffer(4)]],
    index_t height                          [[buffer(5)]],
    index_t width                           [[buffer(6)]],
    index_t kernel_h                        [[buffer(7)]],
    index_t kernel_w                        [[buffer(8)]],
    index_t pad_h                           [[buffer(9)]],
    index_t pad_w                           [[buffer(10)]],
    index_t stride_h                        [[buffer(11)]],
    index_t stride_w                        [[buffer(12)]],
    index_t dilation_h                      [[buffer(13)]],
    index_t dilation_w                      [[buffer(14)]],
    index_t batch_sz                        [[buffer(15)]], //parallel_imgs
    index_t n_offset_grps                   [[buffer(16)]],
    index_t out_h                           [[buffer(17)]],
    index_t out_w                           [[buffer(18)]],
    constant bool & use_mask                [[buffer(19)]],
    constant scalar_t* grad_im              [[buffer(20)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]){
    const index_t grad_im_numel = width * height * channels * batch_sz;

  MPS_1D_KERNEL_LOOP(index, n, 1) {
    const index_t out_x = index % out_w;
    const index_t out_y = (index / out_w) % out_h;
    const index_t b = (index / (out_w * out_h)) % batch_sz;
    const index_t j = (index / (out_w * out_h * batch_sz)) % kernel_w;
    const index_t i =
        (index / (out_w * out_h * batch_sz * kernel_w)) % kernel_h;
    const index_t c = index / (out_w * out_h * batch_sz * kernel_w * kernel_h);

    index_t c_per_offset_grp = channels / n_offset_grps;
    const index_t offset_grp = c / c_per_offset_grp;

    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * kernel_h * kernel_w *
        out_h * out_w;

    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * kernel_h * kernel_w *
          out_h * out_w;
    }

    const index_t mask_idx = i * kernel_w + j;
    const index_t offset_idx = 2 * mask_idx;

    const index_t offset_h_ptr = ((offset_idx)*out_h + out_y) * out_w + out_x;
    const index_t offset_w_ptr =
        ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

    const scalar_t offset_h = offset_ptr[offset_h_ptr];
    const scalar_t offset_w = offset_ptr[offset_w_ptr];

    scalar_t mask_value = 1;
    if (use_mask) {
      mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
    }

    const scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
    const scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

    for (index_t dy = -1; dy <= 1; dy++) {
      for (index_t dx = -1; dx <= 1; dx++) {
        index_t yp = (index_t)y + dy;
        index_t xp = (index_t)x + dx;
        if (0 <= yp && yp < height && 0 <= xp && xp < width &&
            abs(y - yp) < 1 && abs(x - xp) < 1) {
          index_t grad_pos = ((b * channels + c) * height + yp) * width + xp;
          scalar_t weight = (1 - abs(y - yp)) * (1 - abs(x - xp));
            // MSL doesn't support at::native::fastAtomicAdd
            if (grad_pos >= 0 && grad_pos < grad_im_numel) {
                // Atomically add the computed value directly
                atomic_add_float(grad_im + grad_pos, static_cast<float>(mask_value * weight * col[index]));
            }
        }
      }
    }
  }
}

#define REGISTER_DEFORMABLE_COL2IM_OP(DTYPE)   \
template                                  \
[[host_name("deformable_col2im_" #DTYPE)]] \
template <typename scalar_t, typename index_t>  \
kernel void deformable_col2im(                  \
    index_t n                               [[buffer(0)]],  \
    constant scalar_t* col                  [[buffer(1)]],  \
    constant scalar_t* offset_ptr           [[buffer(2)]],  \
    constant scalar_t* mask_ptr             [[buffer(3)]],  \
    index_t channels                        [[buffer(4)]],  \
    index_t height                          [[buffer(5)]],  \
    index_t width                           [[buffer(6)]],  \
    index_t kernel_h                        [[buffer(7)]],  \
    index_t kernel_w                        [[buffer(8)]],  \
    index_t pad_h                           [[buffer(9)]],  \
    index_t pad_w                           [[buffer(10)]],  \
    index_t stride_h                        [[buffer(11)]], \
    index_t stride_w                        [[buffer(12)]], \
    index_t dilation_h                      [[buffer(13)]], \
    index_t dilation_w                      [[buffer(14)]], \
    index_t batch_sz                        [[buffer(15)]], \
    index_t n_offset_grps                   [[buffer(16)]], \
    index_t out_h                           [[buffer(17)]], \
    index_t out_w                           [[buffer(18)]], \
    constant bool & use_mask                [[buffer(19)]], \
    constant scalar_t* grad_im              [[buffer(20)]], \
    uint2     tgid   [[threadgroup_position_in_grid]],      \
    uint2     tptg   [[threads_per_threadgroup]],           \
    uint2     tid2   [[thread_position_in_threadgroup]]);



template <typename scalar_t, typename index_t>
kernel void deformable_col2im_coord(
    index_t n                                   [[buffer(0)]],
    constant scalar_t* col_ptr                  [[buffer(1)]],
    constant scalar_t* im_ptr                   [[buffer(2)]], //input
    constant scalar_t* offset_ptr               [[buffer(3)]],
    constant scalar_t* mask_ptr                 [[buffer(4)]],
    index_t channels                            [[buffer(5)]],
    index_t height                              [[buffer(6)]],
    index_t width                               [[buffer(7)]],
    index_t weight_h                            [[buffer(8)]],
    index_t weight_w                            [[buffer(9)]],
    index_t pad_h                               [[buffer(10)]],
    index_t pad_w                               [[buffer(11)]],
    index_t stride_h                            [[buffer(12)]],
    index_t stride_w                            [[buffer(13)]],
    index_t dilation_h                          [[buffer(14)]],
    index_t dilation_w                          [[buffer(15)]],
    index_t batch_sz                            [[buffer(16)]], //parallel_imgs
    index_t offset_channels                     [[buffer(17)]],
    index_t n_offset_grps                       [[buffer(18)]],
    index_t out_h                               [[buffer(19)]],
    index_t out_w                               [[buffer(20)]],
    constant bool & use_mask                    [[buffer(21)]],
    constant scalar_t* grad_offset              [[buffer(22)]],
    constant scalar_t* grad_mask                [[buffer(23)]],
    uint2     tgid   [[threadgroup_position_in_grid]],
    uint2     tptg   [[threads_per_threadgroup]],
    uint2     tid2   [[thread_position_in_threadgroup]]) {
    MPS_1D_KERNEL_LOOP(index, n, 1) {
    scalar_t grad_offset_val = 0;
    scalar_t grad_mask_val = 0;
    index_t w = index % out_w;
    index_t h = (index / out_w) % out_h;
    index_t w_w = (index / (out_w * out_h * 2)) % weight_w;
    index_t w_h = (index / (out_w * out_h * 2 * weight_w)) % weight_h;
    index_t c = (index / (out_w * out_h)) % offset_channels;
    index_t b = index / (out_w * out_h * offset_channels);

    const index_t offset_grp = c / (2 * weight_h * weight_w);
    const index_t col_step = weight_h * weight_w;

    index_t c_per_offset_grp = channels / n_offset_grps;

    col_ptr += offset_grp * c_per_offset_grp * weight_h * weight_w * batch_sz *
        out_w * out_h;
    im_ptr +=
        (b * n_offset_grps + offset_grp) * c_per_offset_grp * height * width;
    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * weight_h * weight_w *
        out_h * out_w;

    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * weight_h * weight_w *
          out_h * out_w;
    }

    const index_t offset_c = c - offset_grp * 2 * weight_h * weight_w;
    const bool is_y_direction = offset_c % 2 == 0;

    const index_t c_bound = c_per_offset_grp * weight_h * weight_w;
    for (index_t col_c = (offset_c / 2); col_c < c_bound; col_c += col_step) {
      const index_t col_pos =
          (((col_c * batch_sz + b) * out_h) + h) * out_w + w;

      index_t out_x = col_pos % out_w;
      index_t out_y = (col_pos / out_w) % out_h;
      index_t j = (col_pos / (out_w * out_h * batch_sz)) % weight_w;
      index_t i = (col_pos / (out_w * out_h * batch_sz * weight_w)) % weight_h;

      const index_t mask_idx = i * weight_w + j;

      const index_t offset_h_ptr =
          (((2 * mask_idx) * out_h + out_y) * out_w + out_x);
      const index_t offset_w_ptr =
          (((2 * mask_idx + 1) * out_h + out_y) * out_w + out_x);
      const scalar_t offset_h = offset_ptr[offset_h_ptr];
      const scalar_t offset_w = offset_ptr[offset_w_ptr];

      scalar_t mask_value = 1;
      if (use_mask) {
        mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
      }

      scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
      scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

      const scalar_t weight =
          get_coordinate_weight(im_ptr, height, width, y, x, is_y_direction);
      grad_offset_val += mask_value * weight * col_ptr[col_pos];

      if (use_mask && is_y_direction) {
        grad_mask_val += col_ptr[col_pos] *
            bilinear_interpolate(im_ptr, height, width, y, x);
      }

      im_ptr += height * width;
    }

    grad_offset[index] = grad_offset_val;

    if (use_mask && is_y_direction) {
      const index_t idx =
          ((((b * n_offset_grps + offset_grp) * weight_h + w_h) * weight_w +
            w_w) *
               out_h +
           h) *
              out_w +
          w;
      grad_mask[idx] = grad_mask_val;
    }
  }
}

#define REGISTER_DEFORMABLE_COL2IM_COORD_OP(DTYPE)   \
template                                  \
[[host_name("deformable_col2im_coord_" #DTYPE)]] \
template <typename scalar_t, typename index_t>  \
kernel void deformable_col2im_coord(                            \
    index_t n                                   [[buffer(0)]],\
    constant scalar_t* col_ptr                  [[buffer(1)]],  \
    constant scalar_t* im_ptr                   [[buffer(2)]],  \
    constant scalar_t* offset_ptr               [[buffer(3)]],  \
    constant scalar_t* mask_ptr                 [[buffer(4)]],  \
    index_t channels                            [[buffer(5)]],  \
    index_t height                              [[buffer(6)]],  \
    index_t width                               [[buffer(7)]],  \
    index_t weight_h                            [[buffer(8)]],  \
    index_t weight_w                            [[buffer(9)]],  \
    index_t pad_h                               [[buffer(10)]],  \
    index_t pad_w                               [[buffer(11)]], \
    index_t stride_h                            [[buffer(12)]], \
    index_t stride_w                            [[buffer(13)]], \
    index_t dilation_h                          [[buffer(14)]], \
    index_t dilation_w                          [[buffer(15)]], \
    index_t batch_sz                            [[buffer(16)]], \
    index_t offset_channels                     [[buffer(17)]], \
    index_t n_offset_grps                       [[buffer(18)]], \
    index_t out_h                               [[buffer(19)]], \
    index_t out_w                               [[buffer(20)]], \
    constant bool & use_mask                    [[buffer(21)]], \
    constant scalar_t* grad_offset              [[buffer(22)]], \
    constant scalar_t* grad_mask                [[buffer(23)]], \
    uint2     tgid   [[threadgroup_position_in_grid]],          \
    uint2     tptg   [[threads_per_threadgroup]],               \
    uint2     tid2   [[thread_position_in_threadgroup]]);
