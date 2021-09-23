#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <THC/THCAtomics.cuh>

#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

template <typename T>
__device__ T bilinear_interpolate(
    const T* input,
    int height,
    int width,
    T y,
    T x,
    int index /* index for debug only*/) {
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
__global__ void ps_roi_align_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    const T* rois,
    int channels_out,
    T* output,
    int* channel_mapping) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c_out, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c_out = (index / pooled_width / pooled_height) % channels_out;
    int n = index / pooled_width / pooled_height / channels_out;

    // (n, c_in, ph, pw) is the associated element in the input
    int c_in = (c_out * pooled_height + ph) * pooled_width + pw;

    // [start, end) interval for spatial sampling
    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

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
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T count = roi_bin_grid_h * roi_bin_grid_w;

    const T* offset_input =
        input + (roi_batch_ind * channels + c_in) * height * width;
    T out_sum = 0;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = hstart +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
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

template <typename T>
__device__ void bilinear_interpolate_gradient(
    int height,
    int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    int index /* index for debug only*/) {
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

template <typename T>
__global__ void ps_roi_align_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const int* channel_mapping,
    int num_rois,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    int channels_out,
    T* grad_input,
    const T* rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, *, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / channels_out;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

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

    int c_in = channel_mapping[index];
    T* grad_input_offset =
        grad_input + (roi_batch_ind * channels + c_in) * height * width;

    // Do not using floor/ceil; this implementation detail is critical
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;

    const T grad_output_this_bin = grad_output[index];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T count = roi_bin_grid_h * roi_bin_grid_w;

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const T y = hstart +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = wstart +
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
          atomicAdd(grad_input_offset + y_low * width + x_low, g1);
          atomicAdd(grad_input_offset + y_low * width + x_high, g2);
          atomicAdd(grad_input_offset + y_high * width + x_low, g3);
          atomicAdd(grad_input_offset + y_high * width + x_high, g4);
        } // if
      } // ix
    } // iy
  }
}

std::tuple<at::Tensor, at::Tensor> ps_roi_align_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio) {
  // Check if input tensors are CUDA tensors
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  TORCH_CHECK(
      rois.size(1) == 5, "Tensor rois should have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ps_roi_align_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::cuda::CUDAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  TORCH_CHECK(
      channels % (pooled_height * pooled_width) == 0,
      "input channels must be a multiple of pooling height * pooling width");
  int channels_out = channels / (pooled_height * pooled_width);

  auto output = at::zeros(
      {num_rois, channels_out, pooled_height, pooled_width}, input.options());
  auto channel_mapping =
      at::zeros(output.sizes(), input.options().dtype(at::kInt));

  auto output_size = output.numel();
  if (output_size == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output, channel_mapping);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ps_roi_align_forward_kernel", [&] {
        ps_roi_align_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            rois_.data_ptr<scalar_t>(),
            channels_out,
            output.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  return std::make_tuple(output, channel_mapping);
}

at::Tensor ps_roi_align_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  // Check if input tensors are CUDA tensors
  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  TORCH_CHECK(
      channel_mapping.is_cuda(), "channel_mapping must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2},
      channel_mapping_t{channel_mapping, "channel_mapping", 3};

  at::CheckedFrom c = "ps_roi_align_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t, channel_mapping_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::cuda::CUDAGuard device_guard(grad.device());

  auto num_rois = rois.size(0);
  auto grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  int channels_out = channels / (pooled_height * pooled_width);

  auto grad_ = grad.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "ps_roi_align_backward_kernel", [&] {
        ps_roi_align_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad_.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            channels_out,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::ps_roi_align"),
      TORCH_FN(ps_roi_align_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_ps_roi_align_backward"),
      TORCH_FN(ps_roi_align_backward_kernel));
}

} // namespace ops
} // namespace vision
