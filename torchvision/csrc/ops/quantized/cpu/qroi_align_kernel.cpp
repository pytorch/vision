#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/affine_quantizer.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/cpu/vec256/vec256.h>

namespace vision {
namespace ops {

namespace {

template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int iy_upper,
    int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

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
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}


template <typename T>
void qroi_align_forward_kernel_impl(
    int n_rois,
    const T* input,
    double & spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const T* rois,
    T* output,
    int64_t output_size) {

  for (int64_t i = 0; i < output_size; i++) {
      output[i].val_ = 0;
  }

  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0].val_;

    // Do not using rounding; this implementation detail is critical
    // T offset = aligned ? (T)0.5 : (T)0.0;
    int offset = 0;  // TODO fix this
    float roi_start_w = offset_rois[1].val_ * spatial_scale - offset;
    float roi_start_h = offset_rois[2].val_ * spatial_scale - offset;
    float roi_end_w = offset_rois[3].val_ * spatial_scale - offset;
    float roi_end_h = offset_rois[4].val_ * spatial_scale - offset;

    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = std::max(roi_width, 1.f);
      roi_height = std::max(roi_height, 1.f);
    }

    float bin_size_h = roi_height / pooled_height;
    float bin_size_w = roi_width / pooled_width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros.
    const int count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<float>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const T* offset_input =
          input + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          float output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<float> pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_input[pc.pos1].val_ +
                  pc.w2 * offset_input[pc.pos2].val_ +
                  pc.w3 * offset_input[pc.pos3].val_ + pc.w4 * offset_input[pc.pos4].val_;

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          output[index] = at::native::requantize_from_int<T>(1.f,  0, (int)output_val);  // TODO: this is wrong need to set scale and zero etc.
        } // for pw
      } // for ph
    } // for c
  } // for n
    
}

at::Tensor qroi_align_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(rois.device().is_cpu(), "rois must be a CPU tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "qroi_align_forward_kernel";
  at::checkAllSameType(c, {input_t, rois_t});

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  // TODO: This should really be initialized to zero, not empty
  at::Tensor output = at::_empty_affine_quantized(
      {num_rois, channels, pooled_height, pooled_width}, input.options());

  if (output.numel() == 0)
    return output;

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_QINT_TYPES(
      input.scalar_type(), "qroi_align_forward_kernel", [&] {
        qroi_align_forward_kernel_impl<scalar_t>(
            num_rois,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            output.numel());
      });
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN(qroi_align_forward_kernel));
}

} // namespace ops
} // namespace vision