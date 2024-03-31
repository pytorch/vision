#include <ATen/ATen.h>
#include <torch/library.h>

#include "../../cpu/roi_align_common.h"

namespace vision {
namespace ops {

namespace {

// BEGIN copy-pasted code from pytorch core
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/affine_quantizer_base.cpp
// We're vendoring the quantize_val() and dequantize_val() functions here. The
// reason is that these functions belong in at::native, which is incompatible
// with android xplat support.

// FIXME: Remove this section once we can use at::native for android xplat
// builds, or when quantize_val() and dequantize_val() aren't in at::native

#ifdef USE_FBGEMM
template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // Internally, fbgemm::Quantize uses std::nearbyint.
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int32_t qvalue;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  qvalue = fbgemm::Quantize<typename T::underlying, false /*LEGACY*/>(
      value,
      static_cast<int32_t>(zero_point),
      static_cast<float>(scale),
      /*result_precision=*/CHAR_BIT * sizeof(typename T::underlying));
  return static_cast<T>(qvalue);
}

template <typename T>
inline float dequantize_val(double scale, int64_t zero_point, T value) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = static_cast<float>(scale);
  qparams.zero_point = static_cast<int32_t>(zero_point);
  return fbgemm::Dequantize<typename T::underlying>(value.val_, qparams);
}
#else // USE_FBGEMM

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif

template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int64_t qvalue;
  constexpr int64_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int64_t qmax = std::numeric_limits<typename T::underlying>::max();
  float inv_scale = 1.0f / static_cast<float>(scale);
  qvalue = static_cast<int64_t>(zero_point + Round(value * inv_scale));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

template <typename T>
float dequantize_val(double scale, int64_t zero_point, T value) {
  // We need to convert the qint8 value to float to ensure the subtraction
  // subexpression returns a float
  return (static_cast<float>(value.val_) - zero_point) * scale;
}
#endif // USE_FBGEMM
// END copy-pasted code from pytorch core

template <typename T>
void qroi_align_forward_kernel_impl(
    int n_rois,
    const at::Tensor& t_input,
    const float& spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const at::Tensor& t_rois,
    T* output) {
  // Don't delete these otherwise the .data_ptr() data might be undefined
  auto t_input_cont = t_input.contiguous();
  auto t_rois_cont = t_rois.contiguous();

  const T* input = t_input_cont.data_ptr<T>();
  int64_t input_zp = t_input.q_zero_point();
  float input_scale = t_input.q_scale();

  const T* rois = t_rois_cont.data_ptr<T>();
  int64_t rois_zp = t_rois.q_zero_point();
  float rois_scale = t_rois.q_scale();

  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    const T* offset_rois = rois + n * 5;

    // FIXME: change this when batches of size > 1 are allowed
    const int roi_batch_ind = 0;

    // Do not using rounding; this implementation detail is critical
    float offset = aligned ? 0.5 : 0.;
    float roi_start_w =
        dequantize_val(rois_scale, rois_zp, offset_rois[1]) * spatial_scale -
        offset;
    float roi_start_h =
        dequantize_val(rois_scale, rois_zp, offset_rois[2]) * spatial_scale -
        offset;
    float roi_end_w =
        dequantize_val(rois_scale, rois_zp, offset_rois[3]) * spatial_scale -
        offset;
    float roi_end_h =
        dequantize_val(rois_scale, rois_zp, offset_rois[4]) * spatial_scale -
        offset;

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
    const float count =
        std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    // we want to precalculate indices and weights shared by all channels,
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
          float sum_w = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              detail::PreCalc<float> pc = pre_calc[pre_calc_index];

              // Optimization: we use the raw values here and we'll dequantize
              // later
              output_val += pc.w1 * offset_input[pc.pos1].val_ +
                  pc.w2 * offset_input[pc.pos2].val_ +
                  pc.w3 * offset_input[pc.pos3].val_ +
                  pc.w4 * offset_input[pc.pos4].val_;
              sum_w += pc.w1 + pc.w2 + pc.w3 + pc.w4;

              pre_calc_index += 1;
            }
          }
          // Dequantize here
          output_val = input_scale * (output_val - (float)input_zp * sum_w);

          output_val /= count; // Average pooling

          output[index] = quantize_val<T>(input_scale, input_zp, output_val);
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
  // The first column of the RoI tensor is an image index, but not all indices
  // are representable depending on the quantization. For example 1, 3, 5...
  // indices can't be represented when qscale is 2. To prevent any bug, we force
  // a batch size of 1 and we ignore the first column
  TORCH_CHECK(
      input.size(0) == 1,
      "Only one image per batch is allowed in roi_align when quantized tensors are passed.");

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
      {num_rois, channels, pooled_height, pooled_width},
      input.options(),
      input.q_scale(),
      input.q_zero_point());

  if (output.numel() == 0)
    return output;

  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qroi_align_forward_kernel", [&] {
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
