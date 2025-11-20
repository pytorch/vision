/*
ROIAlign implementation in CUDA from pytorch framework
(https://github.com/pytorch/vision/tree/master/torchvision/csrc/cuda on Nov 14 2019)

Adapted for additional 3D capability by G. Ramien, DKFZ Heidelberg
*/

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <cstdio>
#include "cuda_helpers.h"

/*-------------- gpu kernels -----------------*/

template <typename T>
__device__ T linear_interpolate(const T xl, const T val_low, const T val_high){

  T val = (val_high - val_low) * xl + val_low;
  return val;
}

template <typename T>
__device__ T trilinear_interpolate(const T* input, const int height, const int width, const int depth,
                T y, T x, T z, const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) {
    // empty
    return 0;
  }
  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;
  if (z <= 0)
    z = 0;

  int y0 = (int)y;
  int x0 = (int)x;
  int z0 = (int)z;
  int y1;
  int x1;
  int z1;

  if (y0 >= height - 1) {
  /*if nearest gridpoint to y on the lower end is on border or border-1, set low, high, mid(=actual point) to border-1*/
    y1 = y0 = height - 1;
    y = (T)y0;
  } else {
    /* y1 is one pixel from y0, y is the actual point somewhere in between  */
    y1 = y0 + 1;
  }
  if (x0 >= width - 1) {
    x1 = x0 = width - 1;
    x = (T)x0;
  } else {
    x1 = x0 + 1;
  }
  if (z0 >= depth - 1) {
    z1 = z0 = depth - 1;
    z = (T)z0;
  } else {
    z1 = z0 + 1;
  }


  // do linear interpolation of x values
  // distance of actual point to lower boundary point, already normalized since x_high - x0 = 1
  T dis = x - x0;
  /*  accessing element b,c,y,x,z in 1D-rolled-out array of a tensor with dimensions (B, C, Y, X, Z):
      tensor[b,c,y,x,z] = arr[ (((b*C+c)*Y+y)*X + x)*Z + z ] = arr[ alpha + (y*X + x)*Z + z ]
      with alpha = batch&channel locator = (b*C+c)*YXZ.
      hence, as current input pointer is already offset by alpha: y,x,z is at input[( y*X + x)*Z + z], where
      X = width, Z = depth.
  */
  T x00 = linear_interpolate(dis, input[(y0*width+ x0)*depth+z0], input[(y0*width+ x1)*depth+z0]);
  T x10 = linear_interpolate(dis, input[(y1*width+ x0)*depth+z0], input[(y1*width+ x1)*depth+z0]);
  T x01 = linear_interpolate(dis, input[(y0*width+ x0)*depth+z1], input[(y0*width+ x1)*depth+z1]);
  T x11 = linear_interpolate(dis, input[(y1*width+ x0)*depth+z1], input[(y1*width+ x1)*depth+z1]);

  // linear interpol of y values = bilinear interpol of f(x,y)
  dis = y - y0;
  T xy0 = linear_interpolate(dis, x00, x10);
  T xy1 = linear_interpolate(dis, x01, x11);

  // linear interpol of z value = trilinear interpol of f(x,y,z)
  dis = z - z0;
  T xyz = linear_interpolate(dis, xy0, xy1);

  return xyz;
}

template <typename T>
__device__ void trilinear_interpolate_gradient(const int height, const int width, const int depth, T y, T x, T z,
    T& g000, T& g001, T& g010, T& g100, T& g011, T& g101, T& g110, T& g111,
    int& x0, int& x1, int& y0, int& y1, int& z0, int&z1, const int index /* index for debug only*/)
{
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) {
    // empty
    g000 = g001 = g010 = g100 = g011 = g101 = g110 = g111 = 0.;
    x0 = x1 = y0 = y1 = z0 = z1 = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;
  if (z <= 0)
    z = 0;

  y0 = (int)y;
  x0 = (int)x;
  z0 = (int)z;

  if (y0 >= height - 1) {
    y1 = y0 = height - 1;
    y = (T)y0;
  } else {
    y1 = y0 + 1;
  }

  if (x0 >= width - 1) {
    x1 = x0 = width - 1;
    x = (T)x0;
  } else {
    x1 = x0 + 1;
  }

  if (z0 >= depth - 1) {
    z1 = z0 = depth - 1;
    z = (T)z0;
  } else {
    z1 = z0 + 1;
  }

  // forward calculations are added as hints
  T dis_x = x - x0;
  //T x00 = linear_interpolate(dis, input[(y0*width+ x0)*depth+z0], input[(y0*width+ x1)*depth+z0]); // v000, v100
  //T x10 = linear_interpolate(dis, input[(y1*width+ x0)*depth+z0], input[(y1*width+ x1)*depth+z0]); // v010, v110
  //T x01 = linear_interpolate(dis, input[(y0*width+ x0)*depth+z1], input[(y0*width+ x1)*depth+z1]); // v001, v101
  //T x11 = linear_interpolate(dis, input[(y1*width+ x0)*depth+z1], input[(y1*width+ x1)*depth+z1]); // v011, v111

  // linear interpol of y values = bilinear interpol of f(x,y)
  T dis_y = y - y0;
  //T xy0 = linear_interpolate(dis, x00, x10);
  //T xy1 = linear_interpolate(dis, x01, x11);

  // linear interpol of z value = trilinear interpol of f(x,y,z)
  T dis_z = z - z0;
  //T xyz = linear_interpolate(dis, xy0, xy1);

  /* need: grad_i := d(xyz)/d(v_i) with v_i = input_value_i  for all i = 0,..,7 (eight input values --> eight-entry gradient)
     d(lin_interp(dis,x,y))/dx = (-dis +1) and d(lin_interp(dis,x,y))/dy = dis --> derivatives are indep of x,y.
     notation: gxyz = gradient for d(trilin_interp)/d(input_value_at_xyz)
     below grads were calculated by hand
     save time by reusing (1-dis_x) = 1-x+x0 = x1-x =: dis_x1 */
  T dis_x1 = (1-dis_x), dis_y1 = (1-dis_y), dis_z1 = (1-dis_z);

  g000 = dis_z1 * dis_y1  * dis_x1;
  g001 = dis_z  * dis_y1  * dis_x1;
  g010 = dis_z1 * dis_y   * dis_x1;
  g100 = dis_z1 * dis_y1  * dis_x;
  g011 = dis_z  * dis_y   * dis_x1;
  g101 = dis_z  * dis_y1  * dis_x;
  g110 = dis_z1 * dis_y   * dis_x;
  g111 = dis_z  * dis_y   * dis_x;

  return;
}

template <typename T>
__global__ void RoIAlignForward(const int nthreads, const T* input, const T spatial_scale, const int channels,
    const int height, const int width, const int depth, const int pooled_height, const int pooled_width,
    const int pooled_depth, const int sampling_ratio, const bool aligned, const T* rois, T* output)
{

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw, pd) is an element in the pooled output
    int pd =  index % pooled_depth;
    int pw = (index / pooled_depth) % pooled_width;
    int ph = (index / pooled_depth / pooled_width) % pooled_height;
    int c  = (index / pooled_depth / pooled_width / pooled_height) % channels;
    int n  =  index / pooled_depth / pooled_width / pooled_height / channels;


    // rois are (x1,y1,x2,y2,z1,z2) --> tensor of shape (n_rois, 6)
    const T* offset_rois = rois + n * 7;
    int roi_batch_ind = offset_rois[0];
    // aligned==False means legacy version, True means pixel shift by -0.5.
    T offset = aligned ? (T)0.5 : (T)0.0;
    // Do not use rounding; this implementation detail is critical
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;
    T roi_start_d = offset_rois[5] * spatial_scale - offset;
    T roi_end_d = offset_rois[6] * spatial_scale - offset;

    // Force malformed ROIs to be 1x1
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_depth = max(roi_end_d - roi_start_d, (T)1.);

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width * depth;

    // We use roi_bin_grid to sample the grid and mimic integral
    // roi_bin_grid == nr of sampling points per bin >= 1
    int roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    int roi_bin_grid_d =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);

    // We do average (integral) pooling inside a bin
    const T n_voxels = roi_bin_grid_h * roi_bin_grid_w * roi_bin_grid_d; // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5, always in the middle of two grid pointsk

      for (int ix = 0; ix < roi_bin_grid_w; ix++)
      {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        for (int iz = 0; iz < roi_bin_grid_d; iz++)
        {
          const T z = roi_start_d + pd * bin_size_d +
              static_cast<T>(iz + .5f) * bin_size_d / static_cast<T>(roi_bin_grid_d);
          T val = trilinear_interpolate(offset_input, height, width, depth, y, x, z, index);
          output_val += val;
        } // z iterator and calc+add value
      } // x iterator
    } // y iterator
    output_val /= n_voxels;

    output[index] = output_val;
  }
}

template <typename T>
__global__ void RoIAlignBackward(const int nthreads, const T* grad_output, const T spatial_scale, const int channels,
    const int height, const int width, const int depth, const int pooled_height, const int pooled_width,
    const int pooled_depth, const int sampling_ratio, const bool aligned, T* grad_input, const T* rois,
    const int n_stride, const int c_stride, const int h_stride, const int w_stride, const int d_stride)
{

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw, pd) is an element in the pooled output
    int pd =  index % pooled_depth;
    int pw = (index / pooled_depth) % pooled_width;
    int ph = (index / pooled_depth / pooled_width) % pooled_height;
    int c  = (index / pooled_depth / pooled_width / pooled_height) % channels;
    int n  =  index / pooled_depth / pooled_width / pooled_height / channels;


    const T* offset_rois = rois + n * 7;
    int roi_batch_ind = offset_rois[0];
    // aligned==False means legacy version, True means pixel shift by -0.5.
    T offset = aligned ? (T)0.5 : (T)0.0;
    // rois are (x1,y1,x2,y2,z1,z2) --> tensor of shape (n_rois, 6)
    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;
    T roi_start_d = offset_rois[5] * spatial_scale - offset;
    T roi_end_d = offset_rois[6] * spatial_scale - offset;


    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T roi_depth = max(roi_end_d - roi_start_d, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);

    // offset: index b,c,y,x,z of tensor of shape (B,C,Y,X,Z) is
    // b*C*Y*X*Z + c * Y*X*Z + y * X*Z + x *Z + z = (b*C+c)Y*X*Z + ...
    T* offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width * depth);

    // We need to index the gradient using the tensor strides to access the correct values.
    int output_offset = n * n_stride + c * c_stride;
    const T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin = offset_grad_output[ph * h_stride + pw * w_stride + pd * d_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    int roi_bin_grid_d = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);

    // We do average (integral) pooling inside a bin
    const T n_voxels = roi_bin_grid_h * roi_bin_grid_w * roi_bin_grid_d; // e.g. = 6

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5

      for (int ix = 0; ix < roi_bin_grid_w; ix++)
      {
        const T x = roi_start_w + pw * bin_size_w +
          static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

        for (int iz = 0; iz < roi_bin_grid_d; iz++)
        {
          const T z = roi_start_d + pd * bin_size_d +
              static_cast<T>(iz + .5f) * bin_size_d / static_cast<T>(roi_bin_grid_d);

          T g000, g001, g010, g100, g011, g101, g110, g111; // will hold the current partial derivatives
          int x0, x1, y0, y1, z0, z1;
          /* notation: gxyz = gradient at xyz, where x,y,z need to lie on feature-map grid (i.e., =x0,x1 etc.) */
          trilinear_interpolate_gradient(height, width, depth, y, x, z,
                                         g000, g001, g010, g100, g011, g101, g110, g111,
                                         x0, x1, y0, y1, z0, z1, index);
          /* chain rule: derivatives (i.e., the gradient) of trilin_interpolate(v1,v2,v3,v4,...) (div by n_voxels
             as we actually need gradient of whole roi_align) are multiplied with gradient so far*/
          g000 *= grad_output_this_bin / n_voxels;
          g001 *= grad_output_this_bin / n_voxels;
          g010 *= grad_output_this_bin / n_voxels;
          g100 *= grad_output_this_bin / n_voxels;
          g011 *= grad_output_this_bin / n_voxels;
          g101 *= grad_output_this_bin / n_voxels;
          g110 *= grad_output_this_bin / n_voxels;
          g111 *= grad_output_this_bin / n_voxels;

          if (x0 >= 0 && x1 >= 0 && y0 >= 0 && y1 >= 0 && z0 >= 0 && z1 >= 0)
          { // atomicAdd(address, content) reads content under address, adds content to it, while: no other thread
            // can interfere with the memory at address during this operation (thread lock, therefore "atomic").
            atomicAdd(offset_grad_input + (y0 * width + x0) * depth + z0, static_cast<T>(g000));
            atomicAdd(offset_grad_input + (y0 * width + x0) * depth + z1, static_cast<T>(g001));
            atomicAdd(offset_grad_input + (y1 * width + x0) * depth + z0, static_cast<T>(g010));
            atomicAdd(offset_grad_input + (y0 * width + x1) * depth + z0, static_cast<T>(g100));
            atomicAdd(offset_grad_input + (y1 * width + x0) * depth + z1, static_cast<T>(g011));
            atomicAdd(offset_grad_input + (y0 * width + x1) * depth + z1, static_cast<T>(g101));
            atomicAdd(offset_grad_input + (y1 * width + x1) * depth + z0, static_cast<T>(g110));
            atomicAdd(offset_grad_input + (y1 * width + x1) * depth + z1, static_cast<T>(g111));
          } // if
        } // iz
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward


/*----------- wrapper functions ----------------*/

at::Tensor ROIAlign_forward_cuda(const at::Tensor& input, const at::Tensor& rois, const float spatial_scale,
                                const int pooled_height, const int pooled_width, const int pooled_depth,
                                const int sampling_ratio, const bool aligned) {
  /*
   input: feature-map tensor, shape (batch, n_channels, y, x(, z))
   */
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.device().is_cuda(), "rois must be a CUDA tensor");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ROIAlign_forward_cuda";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::cuda::CUDAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);
  auto depth = input.size(4);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width, pooled_depth}, input.options());

  auto output_size = num_rois * channels * pooled_height * pooled_width * pooled_depth;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(static_cast<int64_t>(output_size), static_cast<int64_t>(512)), static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "ROIAlign forward in 3d", [&] {
    RoIAlignForward<scalar_t><<<grid, block, 0, stream>>>(
        output_size,
        input.contiguous().data_ptr<scalar_t>(),
        spatial_scale,
        channels,
        height,
        width,
        depth,
        pooled_height,
        pooled_width,
        pooled_depth,
        sampling_ratio,
        aligned,
        rois.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

at::Tensor ROIAlign_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int depth,
    const int sampling_ratio,
    const bool aligned)
{
  AT_ASSERTM(grad.device().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.device().is_cuda(), "rois must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ROIAlign_backward_cuda";
  at::checkAllSameGPU(c, {grad_t, rois_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::cuda::CUDAGuard device_guard(grad.device());

  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width, depth}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);
  int d_stride = grad.stride(4);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "ROIAlign backward 3D", [&] {
    RoIAlignBackward<scalar_t><<<grid, block, 0, stream>>>(
        grad.numel(),
        grad.data_ptr<scalar_t>(),
        spatial_scale,
        channels,
        height,
        width,
        depth,
        pooled_height,
        pooled_width,
        pooled_depth,
        sampling_ratio,
        aligned,
        grad_input.data_ptr<scalar_t>(),
        rois.contiguous().data_ptr<scalar_t>(),
        n_stride,
        c_stride,
        h_stride,
        w_stride,
        d_stride);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}