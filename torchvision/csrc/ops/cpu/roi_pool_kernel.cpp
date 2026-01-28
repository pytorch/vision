#include <float.h>

#include "../../StableABICompat.h"
#include <torch/csrc/stable/library.h>

namespace vision {
namespace ops {

namespace {

using namespace vision::stable;

template <class T>
inline void add(T* address, const T& val) {
  *address += val;
}

template <typename T>
void roi_pool_forward_kernel_impl(
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const T* rois,
    int num_rois,
    T* output,
    int* argmax_data) {
  for (int n = 0; n < num_rois; ++n) {
    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = round(offset_rois[1] * spatial_scale);
    int roi_start_h = round(offset_rois[2] * spatial_scale);
    int roi_end_w = round(offset_rois[3] * spatial_scale);
    int roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = static_cast<int>(std::floor(static_cast<T>(ph) * bin_size_h));
        int wstart = static_cast<int>(std::floor(static_cast<T>(pw) * bin_size_w));
        int hend = static_cast<int>(std::ceil(static_cast<T>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(std::ceil(static_cast<T>(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = std::min(std::max(hstart + roi_start_h, 0), height);
        hend = std::min(std::max(hend + roi_start_h, 0), height);
        wstart = std::min(std::max(wstart + roi_start_w, 0), width);
        wend = std::min(std::max(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        for (int c = 0; c < channels; ++c) {
          // Define an empty pooling region to be zero
          T maxval = is_empty ? 0 : -FLT_MAX;
          // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
          int maxidx = -1;

          const T* input_offset =
              input + (roi_batch_ind * channels + c) * height * width;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int input_index = h * width + w;
              if (input_offset[input_index] > maxval) {
                maxval = input_offset[input_index];
                maxidx = input_index;
              }
            }
          }
          int index =
              ((n * channels + c) * pooled_height + ph) * pooled_width + pw;
          output[index] = maxval;
          argmax_data[index] = maxidx;
        } // channels
      } // pooled_width
    } // pooled_height
  } // num_rois
}

template <typename T>
void roi_pool_backward_kernel_impl(
    const T* grad_output,
    const int* argmax_data,
    int num_rois,
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
    int w_stride) {
  for (int n = 0; n < num_rois; ++n) {
    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    for (int c = 0; c < channels; ++c) {
      T* grad_input_offset =
          grad_input + ((roi_batch_ind * channels + c) * height * width);
      const int* argmax_data_offset =
          argmax_data + (n * channels + c) * pooled_height * pooled_width;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int output_offset = n * n_stride + c * c_stride;
          int argmax = argmax_data_offset[ph * pooled_width + pw];

          if (argmax != -1) {
            add(grad_input_offset + argmax,
                static_cast<T>(
                    grad_output
                        [output_offset + ph * h_stride + pw * w_stride]));
          }
        } // pooled_width
      } // pooled_height
    } // channels
  } // num_rois
}

std::tuple<Tensor, Tensor> roi_pool_forward_kernel(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  VISION_CHECK(input.is_cpu(), "input must be a CPU tensor");
  VISION_CHECK(rois.is_cpu(), "rois must be a CPU tensor");
  VISION_CHECK(
      input.scalar_type() == rois.scalar_type(),
      "input and rois must have the same dtype");

  int num_rois = rois.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  Tensor output = zeros(
      {num_rois, channels, pooled_height, pooled_width},
      input.scalar_type(),
      Device(kCPU));
  Tensor argmax = zeros(
      {num_rois, channels, pooled_height, pooled_width},
      kInt,
      Device(kCPU));

  if (output.numel() == 0) {
    return std::make_tuple(output, argmax);
  }

  auto input_ = torch::stable::contiguous(input);
  auto rois_ = torch::stable::contiguous(rois);

  auto dtype = input.scalar_type();
  if (dtype == kFloat) {
    roi_pool_forward_kernel_impl<float>(
        input_.const_data_ptr<float>(),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        rois_.const_data_ptr<float>(),
        num_rois,
        output.mutable_data_ptr<float>(),
        argmax.mutable_data_ptr<int>());
  } else if (dtype == kDouble) {
    roi_pool_forward_kernel_impl<double>(
        input_.const_data_ptr<double>(),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        rois_.const_data_ptr<double>(),
        num_rois,
        output.mutable_data_ptr<double>(),
        argmax.mutable_data_ptr<int>());
  } else {
    VISION_CHECK(false, "roi_pool only supports float and double types");
  }
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
  // Check if input tensors are CPU tensors
  VISION_CHECK(grad.is_cpu(), "grad must be a CPU tensor");
  VISION_CHECK(rois.is_cpu(), "rois must be a CPU tensor");
  VISION_CHECK(argmax.is_cpu(), "argmax must be a CPU tensor");
  VISION_CHECK(
      rois.size(1) == 5, "Tensor rois should have shape as Tensor[K, 5]");
  VISION_CHECK(
      grad.scalar_type() == rois.scalar_type(),
      "grad and rois must have the same dtype");

  auto num_rois = rois.size(0);

  Tensor grad_input = zeros(
      {batch_size, channels, height, width},
      grad.scalar_type(),
      Device(kCPU));

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  // get stride values to ensure indexing into gradients is correct.
  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  auto rois_ = torch::stable::contiguous(rois);

  auto dtype = grad.scalar_type();
  if (dtype == kFloat) {
    roi_pool_backward_kernel_impl<float>(
        grad.const_data_ptr<float>(),
        argmax.const_data_ptr<int>(),
        num_rois,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        grad_input.mutable_data_ptr<float>(),
        rois_.const_data_ptr<float>(),
        n_stride,
        c_stride,
        h_stride,
        w_stride);
  } else if (dtype == kDouble) {
    roi_pool_backward_kernel_impl<double>(
        grad.const_data_ptr<double>(),
        argmax.const_data_ptr<int>(),
        num_rois,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        grad_input.mutable_data_ptr<double>(),
        rois_.const_data_ptr<double>(),
        n_stride,
        c_stride,
        h_stride,
        w_stride);
  } else {
    VISION_CHECK(false, "roi_pool backward only supports float and double types");
  }
  return grad_input;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl("roi_pool", TORCH_BOX(&roi_pool_forward_kernel));
  m.impl("_roi_pool_backward", TORCH_BOX(&roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision
