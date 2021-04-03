#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/affine_quantizer.h>

#include "../../roi_align.h"

namespace vision {
namespace ops {

namespace {

template <typename T>
void qroi_align_forward_kernel_impl(
    int n_rois,
    const at::Tensor& t_input,
    const float & spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const at::Tensor& t_rois,
    T* output){
  
  const T* input = t_input.contiguous().data_ptr<T>();
  const T* rois = t_rois.contiguous().data_ptr<T>();
  
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0].val_; // FIXME: This can be out of the range of the quantized type!!

    // Do not using rounding; this implementation detail is critical
    float offset = aligned ? 0.5 : 0.;
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
    const float count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    // we want to precalculate indices and weights shared by all chanels,
    // this is the key point of optimization
    std::vector<detail::PreCalc<float>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    detail::pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
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
              detail::PreCalc<float> pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_input[pc.pos1].val_ +
                  pc.w2 * offset_input[pc.pos2].val_ +
                  pc.w3 * offset_input[pc.pos3].val_ + pc.w4 * offset_input[pc.pos4].val_;

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          output[index] = at::native::quantize_val<T>(1.f,  0, output_val);  // TODO: this is wrong need to set scale and zero etc.
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

  // FIXME: This is private, API might change:
  // https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor#quantized-tensor-apis
  at::Tensor output = at::_empty_affine_quantized(
      {num_rois, channels, pooled_height, pooled_width}, input.options(),
       input.q_scale(), input.q_zero_point());

  if (output.numel() == 0)
    return output;

  AT_DISPATCH_QINT_TYPES(
      input.scalar_type(), "qroi_align_forward_kernel", [&] {
        qroi_align_forward_kernel_impl<scalar_t>(
            num_rois,
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
            output.data_ptr<scalar_t>());
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
