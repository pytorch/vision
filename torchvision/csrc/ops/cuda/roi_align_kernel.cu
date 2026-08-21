#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/cuda/AtomicAdd.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

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
__global__ void roi_align_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const T* rois,
    T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
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

    const T* offset_input =
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
__global__ void roi_align_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    T* grad_input,
    const T* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride,
    const int memory_span) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
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
    const T* offset_grad_output = grad_output + output_offset;
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
          torch::headeronly::fastAtomicAdd(
              grad_input,
              input_offset + y_low * width + x_low,
              memory_span,
              static_cast<T>(g1),
              true);
          torch::headeronly::fastAtomicAdd(
              grad_input,
              input_offset + y_low * width + x_high,
              memory_span,
              static_cast<T>(g2),
              true);
          torch::headeronly::fastAtomicAdd(
              grad_input,
              input_offset + y_high * width + x_low,
              memory_span,
              static_cast<T>(g3),
              true);
          torch::headeronly::fastAtomicAdd(
              grad_input,
              input_offset + y_high * width + x_high,
              memory_span,
              static_cast<T>(g4),
              true);
        } // if
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
}

// THO_DISPATCH_V2 splits its body on commas outside parens. The commas in
// kernel<<<grid, block, 0, stream>>> would break it, so it goes through this
// wrapper.
template <typename scalar_t>
void launch_roi_align_forward_kernel_impl(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    int output_size,
    const scalar_t* input,
    double spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const scalar_t* rois,
    scalar_t* output) {
  roi_align_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
      output_size,
      input,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned,
      rois,
      output);
}

template <typename scalar_t>
void launch_roi_align_backward_kernel_impl(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    int nthreads,
    const scalar_t* grad_output,
    double spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    scalar_t* grad_input,
    const scalar_t* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride,
    int memory_span) {
  roi_align_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
      nthreads,
      grad_output,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned,
      grad_input,
      rois,
      n_stride,
      c_stride,
      h_stride,
      w_stride,
      memory_span);
}

Tensor roi_align_forward_kernel(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  STD_TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  STD_TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");
  STD_TORCH_CHECK(
      input.get_device_index() == rois.get_device_index(),
      "input should be on the same device as rois");
  STD_TORCH_CHECK(
      input.scalar_type() == rois.scalar_type(),
      "input should have the same type as rois");

  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  Tensor output = torch::stable::new_zeros(
      input, {num_rois, channels, pooled_height, pooled_width});

  auto output_size = num_rois * pooled_height * pooled_width * channels;

  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      input.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    STD_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  auto input_ = torch::stable::contiguous(input);
  auto rois_ = torch::stable::contiguous(rois);
  THO_DISPATCH_V2(
      input.scalar_type(),
      "roi_align_forward_kernel",
      AT_WRAP([&]() {
        launch_roi_align_forward_kernel_impl<scalar_t>(
            grid,
            block,
            stream,
            output_size,
            input_.const_data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            rois_.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

Tensor roi_align_backward_kernel(
    const Tensor& grad,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  STD_TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  STD_TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  STD_TORCH_CHECK(
      grad.get_device_index() == rois.get_device_index(),
      "grad should be on the same device as rois");
  STD_TORCH_CHECK(
      grad.scalar_type() == rois.scalar_type(),
      "grad should have the same type as rois");

  torch::stable::accelerator::DeviceGuard device_guard(grad.get_device_index());

  Tensor grad_input =
      torch::stable::new_zeros(grad, {batch_size, channels, height, width});

  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(grad.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    STD_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  auto rois_ = torch::stable::contiguous(rois);
  THO_DISPATCH_V2(
      grad.scalar_type(),
      "roi_align_backward_kernel",
      AT_WRAP([&]() {
        launch_roi_align_backward_kernel_impl<scalar_t>(
            grid,
            block,
            stream,
            grad.numel(),
            grad.const_data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            grad_input.mutable_data_ptr<scalar_t>(),
            rois_.const_data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride,
            grad_input.numel());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_input;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("roi_align", TORCH_BOX(&roi_align_forward_kernel));
  m.impl("_roi_align_backward", TORCH_BOX(&roi_align_backward_kernel));
}

} // namespace ops
} // namespace vision
