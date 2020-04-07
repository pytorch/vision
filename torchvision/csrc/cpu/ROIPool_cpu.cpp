#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <TH/TH.h>
#include <algorithm>

template <class T>
inline void add(T* address, const T& val) {
  *address += val;
}

template <typename T>
void RoIPoolForward(
    const T* input,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* rois,
    const int num_rois,
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
void RoIPoolBackward(
    const T* grad_output,
    const int* argmax_data,
    const int num_rois,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* grad_input,
    const T* rois,
    const int n_stride,
    const int c_stride,
    const int h_stride,
    const int w_stride) {
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

std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width) {
  AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
  AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ROIPool_forward_cpu";
  at::checkAllSameType(c, {input_t, rois_t});

  int num_rois = rois.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros(
      {num_rois, channels, pooled_height, pooled_width},
      input.options().dtype(at::kInt));

  if (output.numel() == 0) {
    return std::make_tuple(output, argmax);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ROIPool_forward", [&] {
        RoIPoolForward<scalar_t>(
            input.contiguous().data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois.contiguous().data_ptr<scalar_t>(),
            num_rois,
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>());
      });
  return std::make_tuple(output, argmax);
}

at::Tensor ROIPool_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
  // Check if input tensors are CPU tensors
  AT_ASSERTM(grad.device().is_cpu(), "grad must be a CPU tensor");
  AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");
  AT_ASSERTM(argmax.device().is_cpu(), "argmax must be a CPU tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ROIPool_backward_cpu";
  at::checkAllSameType(c, {grad_t, rois_t});

  auto num_rois = rois.size(0);

  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  // get stride values to ensure indexing into gradients is correct.
  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "ROIPool_backward", [&] {
        RoIPoolBackward<scalar_t>(
            grad.data_ptr<scalar_t>(),
            argmax.data_ptr<int>(),
            num_rois,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            grad_input.data_ptr<scalar_t>(),
            rois.contiguous().data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride);
      });
  return grad_input;
}
