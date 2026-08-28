#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/cuda/KernelUtils.h>
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
__global__ void ps_roi_pool_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
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
    int roi_start_w = roundf(offset_rois[1] * spatial_scale);
    int roi_start_h = roundf(offset_rois[2] * spatial_scale);
    int roi_end_w = roundf(offset_rois[3] * spatial_scale);
    int roi_end_h = roundf(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w, 1);
    int roi_height = max(roi_end_h - roi_start_h, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height - 1);
    hend = min(max(hend + roi_start_h, 0), height - 1);
    wstart = min(max(wstart + roi_start_w, 0), width - 1);
    wend = min(max(wend + roi_start_w, 0), width - 1);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    const T* offset_input =
        input + (roi_batch_ind * channels + c_in) * height * width;
    T out_sum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * width + w;
        out_sum += offset_input[input_index];
      }
    }

    T bin_area = (hend - hstart) * (wend - wstart);
    output[index] = is_empty ? static_cast<T>(0) : out_sum / bin_area;
    channel_mapping[index] = c_in;
  }
}

template <typename T>
__global__ void ps_roi_pool_backward_kernel_impl(
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
    int channels_out,
    T* grad_input,
    const T* rois,
    const int memory_span) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, *, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / channels_out;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = roundf(offset_rois[1] * spatial_scale);
    int roi_start_h = roundf(offset_rois[2] * spatial_scale);
    int roi_end_w = roundf(offset_rois[3] * spatial_scale);
    int roi_end_h = roundf(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w, 1);
    int roi_height = max(roi_end_h - roi_start_h, 1);
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

    int c_in = channel_mapping[index];
    T bin_area = (hend - hstart) * (wend - wstart);
    T diff_val = is_empty ? static_cast<T>(0) : grad_output[index] / bin_area;

    const int offset = (roi_batch_ind * channels + c_in) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int grad_input_index = h * width + w;
        torch::headeronly::fastAtomicAdd(
            grad_input, offset + grad_input_index, memory_span, diff_val, true);
      }
    }
  }
}

// THO_DISPATCH_V2 splits its body on commas outside parens. The commas in
// kernel<<<grid, block, 0, stream>>> would break it, so it goes through this
// wrapper.
template <typename scalar_t>
void launch_ps_roi_pool_forward_kernel_impl(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    int nthreads,
    const scalar_t* input,
    const scalar_t spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const scalar_t* rois,
    int channels_out,
    scalar_t* output,
    int* channel_mapping) {
  ps_roi_pool_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
      nthreads,
      input,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      rois,
      channels_out,
      output,
      channel_mapping);
}

template <typename scalar_t>
void launch_ps_roi_pool_backward_kernel_impl(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    int nthreads,
    const scalar_t* grad_output,
    const int* channel_mapping,
    int num_rois,
    const scalar_t spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int channels_out,
    scalar_t* grad_input,
    const scalar_t* rois,
    const int memory_span) {
  ps_roi_pool_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
      nthreads,
      grad_output,
      channel_mapping,
      num_rois,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      channels_out,
      grad_input,
      rois,
      memory_span);
}

std::tuple<Tensor, Tensor> ps_roi_pool_forward_kernel(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  // Check if input tensors are CUDA tensors
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

  STD_TORCH_CHECK(
      channels % (pooled_height * pooled_width) == 0,
      "input channels must be a multiple of pooling height * pooling width");
  int channels_out = channels / (pooled_height * pooled_width);

  Tensor output = torch::stable::new_zeros(
      input, {num_rois, channels_out, pooled_height, pooled_width});
  Tensor channel_mapping = torch::stable::new_zeros(
      input,
      {num_rois, channels_out, pooled_height, pooled_width},
      torch::headeronly::ScalarType::Int);

  auto output_size = output.numel();
  if (output_size == 0) {
    STD_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(output, channel_mapping);
  }

  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  auto input_ = torch::stable::contiguous(input);
  auto rois_ = torch::stable::contiguous(rois);
  THO_DISPATCH_V2(
      input.scalar_type(),
      "ps_roi_pool_forward_kernel",
      AT_WRAP([&]() {
        launch_ps_roi_pool_forward_kernel_impl<scalar_t>(
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
            channels_out,
            output.mutable_data_ptr<scalar_t>(),
            channel_mapping.mutable_data_ptr<int>());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(output, channel_mapping);
}

Tensor ps_roi_pool_backward_kernel(
    const Tensor& grad,
    const Tensor& rois,
    const Tensor& channel_mapping,
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
  STD_TORCH_CHECK(
      channel_mapping.is_cuda(), "channel_mapping must be a CUDA tensor");
  STD_TORCH_CHECK(
      grad.get_device_index() == rois.get_device_index(),
      "grad should be on the same device as rois");
  STD_TORCH_CHECK(
      grad.get_device_index() == channel_mapping.get_device_index(),
      "grad should be on the same device as channel_mapping");
  STD_TORCH_CHECK(
      grad.scalar_type() == rois.scalar_type(),
      "grad should have the same type as rois");

  torch::stable::accelerator::DeviceGuard device_guard(grad.get_device_index());

  auto num_rois = rois.size(0);
  Tensor grad_input =
      torch::stable::new_zeros(grad, {batch_size, channels, height, width});

  cudaStream_t stream = get_current_cuda_stream(grad.get_device_index());

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    STD_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
  }

  int channels_out = channels / (pooled_height * pooled_width);

  auto grad_ = torch::stable::contiguous(grad);
  auto rois_ = torch::stable::contiguous(rois);
  THO_DISPATCH_V2(
      grad.scalar_type(),
      "ps_roi_pool_backward_kernel",
      AT_WRAP([&]() {
        launch_ps_roi_pool_backward_kernel_impl<scalar_t>(
            grid,
            block,
            stream,
            grad.numel(),
            grad_.const_data_ptr<scalar_t>(),
            channel_mapping.const_data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            channels_out,
            grad_input.mutable_data_ptr<scalar_t>(),
            rois_.const_data_ptr<scalar_t>(),
            grad_input.numel());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_input;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl("ps_roi_pool", TORCH_BOX(&ps_roi_pool_forward_kernel));
  m.impl("_ps_roi_pool_backward", TORCH_BOX(&ps_roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision
