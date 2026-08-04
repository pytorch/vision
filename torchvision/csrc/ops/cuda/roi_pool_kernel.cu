#include <float.h>

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
#include <tuple>

#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

template <typename T>
__global__ void roi_pool_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const T* rois,
    T* output,
    int* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = round(offset_rois[1] * spatial_scale);
    int roi_start_h = round(offset_rois[2] * spatial_scale);
    int roi_end_w = round(offset_rois[3] * spatial_scale);
    int roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * width + w;
        if (offset_input[input_index] > maxval) {
          maxval = offset_input[input_index];
          maxidx = input_index;
        }
      }
    }
    output[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename T>
__global__ void roi_pool_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const int* argmax_data,
    int num_rois,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
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

    const int output_offset = n * n_stride + c * c_stride;
    const int* argmax_data_offset =
        argmax_data + (n * channels + c) * pooled_height * pooled_width;
    const int argmax = argmax_data_offset[ph * pooled_width + pw];
    const int offset = (roi_batch_ind * channels + c) * height * width;

    if (argmax != -1) {
      torch::headeronly::fastAtomicAdd(
          grad_input,
          offset + argmax,
          memory_span,
          static_cast<T>(
              grad_output[output_offset + ph * h_stride + pw * w_stride]),
          true);
    }
  }
}

// THO_DISPATCH_V2 splits its body on commas outside parens. The commas in
// kernel<<<grid, block, 0, stream>>> would break it, so it goes through this
// wrapper.
template <typename scalar_t>
void launch_roi_pool_forward_kernel_impl(
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
    const scalar_t* rois,
    scalar_t* output,
    int* argmax_data) {
  roi_pool_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
      output_size,
      input,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      rois,
      output,
      argmax_data);
}

template <typename scalar_t>
void launch_roi_pool_backward_kernel_impl(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    int nthreads,
    const scalar_t* grad_output,
    const int* argmax_data,
    int num_rois,
    double spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    scalar_t* grad_input,
    const scalar_t* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride,
    int memory_span) {
  roi_pool_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
      nthreads,
      grad_output,
      argmax_data,
      num_rois,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      grad_input,
      rois,
      n_stride,
      c_stride,
      h_stride,
      w_stride,
      memory_span);
}

std::tuple<Tensor, Tensor> roi_pool_forward_kernel(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  STD_TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  STD_TORCH_CHECK(
      rois.size(1) == 5, "Tensor rois should have shape as Tensor[K, 5]");
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
  Tensor argmax = torch::stable::new_zeros(
      input,
      {num_rois, channels, pooled_height, pooled_width},
      torch::headeronly::ScalarType::Int);

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
    return std::make_tuple(output, argmax);
  }

  auto input_ = torch::stable::contiguous(input);
  auto rois_ = torch::stable::contiguous(rois);
  THO_DISPATCH_V2(
      input.scalar_type(),
      "roi_pool_forward_kernel",
      AT_WRAP([&]() {
        launch_roi_pool_forward_kernel_impl<scalar_t>(
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
            rois_.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            argmax.mutable_data_ptr<int>());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(output, argmax);
}

Tensor roi_pool_backward_kernel(
    const Tensor& grad,
    const Tensor& rois,
    const Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  // Check if input tensors are CUDA tensors
  STD_TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  STD_TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  STD_TORCH_CHECK(argmax.is_cuda(), "argmax must be a CUDA tensor");
  STD_TORCH_CHECK(
      grad.get_device_index() == rois.get_device_index(),
      "grad should be on the same device as rois");
  STD_TORCH_CHECK(
      grad.get_device_index() == argmax.get_device_index(),
      "grad should be on the same device as argmax");
  STD_TORCH_CHECK(
      grad.scalar_type() == rois.scalar_type(),
      "grad should have the same type as rois");

  torch::stable::accelerator::DeviceGuard device_guard(grad.get_device_index());

  auto num_rois = rois.size(0);

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

  auto argmax_ = torch::stable::contiguous(argmax);
  auto rois_ = torch::stable::contiguous(rois);
  THO_DISPATCH_V2(
      grad.scalar_type(),
      "roi_pool_backward_kernel",
      AT_WRAP([&]() {
        launch_roi_pool_backward_kernel_impl<scalar_t>(
            grid,
            block,
            stream,
            grad.numel(),
            grad.const_data_ptr<scalar_t>(),
            argmax_.const_data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
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
  m.impl("roi_pool", TORCH_BOX(&roi_pool_forward_kernel));
  m.impl("_roi_pool_backward", TORCH_BOX(&roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision
