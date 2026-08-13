#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>

#include <algorithm>
#include <cmath>
#include <tuple>

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

template <class T>
inline void add(T* address, const T& val) {
  *address += val;
}

template <typename T>
void ps_roi_pool_forward_kernel_impl(
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const T* rois,
    int channels_out,
    int num_rois,
    T* output,
    int* channel_mapping) {
  for (int n = 0; n < num_rois; ++n) {
    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = round(offset_rois[1] * spatial_scale);
    int roi_start_h = round(offset_rois[2] * spatial_scale);
    int roi_end_w = round(offset_rois[3] * spatial_scale);
    int roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    int roi_width = std::max(roi_end_w - roi_start_w, 1);
    int roi_height = std::max(roi_end_h - roi_start_h, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int c_in = 0;
    for (int c_out = 0; c_out < channels_out; ++c_out) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
          int hend =
              static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
          int wend =
              static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

          // Add roi offsets and clip to input boundaries
          hstart = std::min(std::max(hstart + roi_start_h, 0), height - 1);
          hend = std::min(std::max(hend + roi_start_h, 0), height - 1);
          wstart = std::min(std::max(wstart + roi_start_w, 0), width - 1);
          wend = std::min(std::max(wend + roi_start_w, 0), width - 1);
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

          int index =
              ((n * channels_out + c_out) * pooled_height + ph) * pooled_width +
              pw;
          T bin_area = (hend - hstart) * (wend - wstart);
          output[index] = is_empty ? static_cast<T>(0) : out_sum / bin_area;
          channel_mapping[index] = c_in;
          c_in++;
        }
      }
    }
  }
}

template <typename T>
void ps_roi_pool_backward_kernel_impl(
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
    const T* rois) {
  for (int n = 0; n < num_rois; ++n) {
    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = roundf(offset_rois[1] * spatial_scale);
    int roi_start_h = roundf(offset_rois[2] * spatial_scale);
    int roi_end_w = roundf(offset_rois[3] * spatial_scale);
    int roi_end_h = roundf(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    int roi_width = std::max(roi_end_w - roi_start_w, 1);
    int roi_height = std::max(roi_end_h - roi_start_h, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = std::min(std::max(hstart + roi_start_h, 0), height);
        hend = std::min(std::max(hend + roi_start_h, 0), height);
        wstart = std::min(std::max(wstart + roi_start_w, 0), width);
        wend = std::min(std::max(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        for (int c_out = 0; c_out < channels_out; ++c_out) {
          int index =
              ((n * channels_out + c_out) * pooled_height + ph) * pooled_width +
              pw;
          int c_in = channel_mapping[index];

          T* grad_input_offset =
              grad_input + (roi_batch_ind * channels + c_in) * height * width;
          T bin_area = (hend - hstart) * (wend - wstart);
          T diff_val =
              is_empty ? static_cast<T>(0) : grad_output[index] / bin_area;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int grad_input_index = h * width + w;
              add(grad_input_offset + grad_input_index, diff_val);
            }
          }
        }
      }
    }
  }
}

std::tuple<Tensor, Tensor> ps_roi_pool_forward_kernel(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  // Check if input tensors are CPU tensors
  STD_TORCH_CHECK(input.is_cpu(), "input must be a CPU tensor");
  STD_TORCH_CHECK(rois.is_cpu(), "rois must be a CPU tensor");
  STD_TORCH_CHECK(
      rois.size(1) == 5, "Tensor rois should have shape as Tensor[K, 5]");
  STD_TORCH_CHECK(
      input.scalar_type() == rois.scalar_type(),
      "input should have the same type as rois");

  int num_rois = rois.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

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
    return std::make_tuple(output, channel_mapping);
  }

  auto input_ = torch::stable::contiguous(input);
  auto rois_ = torch::stable::contiguous(rois);
  THO_DISPATCH_V2(
      input.scalar_type(),
      "ps_roi_pool_forward_kernel",
      AT_WRAP([&]() {
        ps_roi_pool_forward_kernel_impl<scalar_t>(
            input_.const_data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.const_data_ptr<scalar_t>(),
            channels_out,
            num_rois,
            output.mutable_data_ptr<scalar_t>(),
            channel_mapping.mutable_data_ptr<int>());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
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
  // Check if input tensors are CPU tensors
  STD_TORCH_CHECK(grad.is_cpu(), "grad must be a CPU tensor");
  STD_TORCH_CHECK(rois.is_cpu(), "rois must be a CPU tensor");
  STD_TORCH_CHECK(
      channel_mapping.is_cpu(), "channel_mapping must be a CPU tensor");
  STD_TORCH_CHECK(
      grad.scalar_type() == rois.scalar_type(),
      "grad should have the same type as rois");

  auto num_rois = rois.size(0);
  Tensor grad_input =
      torch::stable::new_zeros(grad, {batch_size, channels, height, width});

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  int channels_out = channels / (pooled_height * pooled_width);

  auto grad_ = torch::stable::contiguous(grad);
  auto rois_ = torch::stable::contiguous(rois);
  THO_DISPATCH_V2(
      grad.scalar_type(),
      "ps_roi_pool_backward_kernel",
      AT_WRAP([&]() {
        ps_roi_pool_backward_kernel_impl<scalar_t>(
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
            rois_.const_data_ptr<scalar_t>());
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      torch::headeronly::ScalarType::Half);
  return grad_input;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("ps_roi_pool", TORCH_BOX(&ps_roi_pool_forward_kernel));
  m.impl("_ps_roi_pool_backward", TORCH_BOX(&ps_roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision
