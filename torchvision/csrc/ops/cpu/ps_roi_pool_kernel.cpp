#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

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

std::tuple<at::Tensor, at::Tensor> ps_roi_pool_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  // Check if input tensors are CPU tensors
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(rois.device().is_cpu(), "rois must be a CPU tensor");
  TORCH_CHECK(
      rois.size(1) == 5, "Tensor rois should have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ps_roi_pool_forward_kernel";
  at::checkAllSameType(c, {input_t, rois_t});

  int num_rois = rois.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

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
    return std::make_tuple(output, channel_mapping);
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ps_roi_pool_forward_kernel", [&] {
        ps_roi_pool_forward_kernel_impl<scalar_t>(
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.data_ptr<scalar_t>(),
            channels_out,
            num_rois,
            output.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>());
      });
  return std::make_tuple(output, channel_mapping);
}

at::Tensor ps_roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  // Check if input tensors are CPU tensors
  TORCH_CHECK(grad.device().is_cpu(), "grad must be a CPU tensor");
  TORCH_CHECK(rois.device().is_cpu(), "rois must be a CPU tensor");
  TORCH_CHECK(
      channel_mapping.device().is_cpu(),
      "channel_mapping must be a CPU tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2},
      channel_mapping_t{channel_mapping, "channel_mapping", 3};

  at::CheckedFrom c = "ps_roi_pool_backward_kernel";
  at::checkAllSameType(c, {grad_t, rois_t});

  auto num_rois = rois.size(0);
  auto grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  int channels_out = channels / (pooled_height * pooled_width);

  auto grad_ = grad.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "ps_roi_pool_backward_kernel", [&] {
        ps_roi_pool_backward_kernel_impl<scalar_t>(
            grad_.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            channels_out,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>());
      });
  return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::ps_roi_pool"),
      TORCH_FN(ps_roi_pool_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_ps_roi_pool_backward"),
      TORCH_FN(ps_roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision
