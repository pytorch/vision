#include <ATen/Parallel.h>
#include <ATen/TypeDefault.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UpSample.h>
#include <cmath>
#include <vector>

#include <torch/library.h>

// Code temporary is in torchvision before merging it to PyTorch
namespace at {
namespace native {
namespace internal_upsample {

using scale_t = std::vector<c10::optional<double>>;

template <typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa_single_dim_zero_strides(
    char* src,
    char** data,
    int64_t i,
    const index_t ids_stride) {
  const index_t ids_min = *(index_t*)&data[0][0];
  const index_t ids_size = *(index_t*)&data[1][0];

  char* src_min = src + ids_min;

  scalar_t t = *(scalar_t*)&src_min[0];
  index_t wts_idx = *(index_t*)&data[4][0];
  scalar_t* wts_ptr = (scalar_t*)&data[3][wts_idx];
  scalar_t wts = wts_ptr[0];

  scalar_t output = t * wts;
  int j = 1;
  for (; j < ids_size; j++) {
    wts = wts_ptr[j];
    t = *(scalar_t*)&src_min[j * ids_stride];
    output += t * wts;
  }
  return output;
}

template <typename scalar_t, typename index_t>
static inline scalar_t interpolate_aa_single_dim(
    char* src,
    char** data,
    const int64_t* strides,
    int64_t i,
    const index_t ids_stride) {
  index_t ids_min = *(index_t*)&data[0][i * strides[0]];
  index_t ids_size = *(index_t*)&data[1][i * strides[1]];

  char* src_min = src + ids_min;

  scalar_t t = *(scalar_t*)&src_min[0];
  index_t wts_idx = *(index_t*)&data[4][i * strides[4]];
  scalar_t* wts_ptr = (scalar_t*)&data[3][wts_idx];
  scalar_t wts = wts_ptr[0];

  scalar_t output = t * wts;
  int j = 1;
  for (; j < ids_size; j++) {
    wts = wts_ptr[j];
    t = *(scalar_t*)&src_min[j * ids_stride];
    output += t * wts;
  }
  return output;
}

template <typename scalar_t, typename index_t>
static inline void basic_loop_aa_single_dim_zero_strides(
    char** data,
    const int64_t* strides,
    int64_t n) {
  char* dst = data[0];
  char* src = data[1];
  // index stride is constant for the given dimension
  const index_t ids_stride = *(index_t*)&data[2 + 2][0];

  for (int64_t i = 0; i < n; i++) {
    *(scalar_t*)&dst[i * strides[0]] =
        interpolate_aa_single_dim_zero_strides<scalar_t, index_t>(
            src + i * strides[1], &data[2], i, ids_stride);
  }
}

template <typename scalar_t, typename index_t>
static inline void basic_loop_aa_single_dim_nonzero_strides(
    char** data,
    const int64_t* strides,
    int64_t n) {
  char* dst = data[0];
  char* src = data[1];
  // index stride is constant for the given dimension
  const index_t ids_stride = *(index_t*)&data[2 + 2][0];

  if (strides[1] == 0) {
    for (int64_t i = 0; i < n; i++) {
      *(scalar_t*)&dst[i * strides[0]] =
          interpolate_aa_single_dim<scalar_t, index_t>(
              src, &data[2], &strides[2], i, ids_stride);
    }
  } else {
    for (int64_t i = 0; i < n; i++) {
      *(scalar_t*)&dst[i * strides[0]] =
          interpolate_aa_single_dim<scalar_t, index_t>(
              src + i * strides[1], &data[2], &strides[2], i, ids_stride);
    }
  }
}

template <int m>
static inline bool is_zero_stride(const int64_t* strides) {
  bool output = strides[0] == 0;
  for (int i = 1; i < m; i++) {
    output &= (strides[i] == 0);
  }
  return output;
}

template <typename scalar_t, typename index_t, int out_ndims>
void ti_cpu_upsample_generic_aa(
    at::TensorIterator& iter,
    int interp_size = -1) {
  TORCH_INTERNAL_ASSERT(interp_size > 0);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    if ((strides[0] == sizeof(scalar_t)) && (strides[1] == sizeof(scalar_t)) &&
        is_zero_stride<3 + 2>(&strides[2])) {
      basic_loop_aa_single_dim_zero_strides<scalar_t, index_t>(
          data, strides, n);
    } else {
      basic_loop_aa_single_dim_nonzero_strides<scalar_t, index_t>(
          data, strides, n);
    }
  };

  iter.for_each(loop);
}

// Helper structs to use with ti_upsample_generic_Nd_kernel_impl
template <typename index_t, typename scalar_t>
struct HelperInterpBase {
  template <typename filter_fn_t>
  static inline void _compute_weights_aa(
      const int64_t i,
      const int64_t input_size,
      const scalar_t scale,
      const scalar_t support,
      scalar_t* wt_ptr,
      const int64_t interp_size,
      filter_fn_t filter_fn,
      int64_t& xmin,
      int64_t& xsize) {
    scalar_t center = scale * (i + 0.5);
    scalar_t total_w = 0.0;
    scalar_t invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
    xmin = std::max(
        static_cast<int64_t>(center - support + 0.5), static_cast<index_t>(0));
    xsize = std::min(static_cast<int64_t>(center + support + 0.5), input_size) -
        xmin;

    int64_t j = 0;
    for (; j < xsize; j++) {
      scalar_t w = filter_fn((j + xmin - center + 0.5) * invscale);
      wt_ptr[j] = w;
      total_w += w;
    }
    for (j = 0; j < xsize; j++) {
      if (total_w != 0.0) {
        wt_ptr[j] /= total_w;
      }
    }
    for (; j < interp_size; j++) {
      wt_ptr[j] = static_cast<scalar_t>(0.0);
    }
  }

  template <typename filter_fn_t>
  static inline std::vector<Tensor> _compute_indices_weights_aa(
      int64_t input_size,
      int64_t output_size,
      int64_t stride,
      int64_t ndims,
      int64_t reshape_dim,
      bool align_corners,
      scalar_t scale,
      int& in_out_interp_size,
      filter_fn_t filter_fn) {
    int interp_size = in_out_interp_size;
    scalar_t support =
        (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
    interp_size = (int)ceilf(support) * 2 + 1;

    // return interp_size
    in_out_interp_size = interp_size;

    std::vector<Tensor> output;
    auto new_shape = std::vector<int64_t>(ndims, 1);
    new_shape[reshape_dim] = output_size;

    // ---- Bounds approach as in PIL -----
    // bounds: xmin/xmax
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
    output.emplace_back(
        empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));

    {
      // Weights
      new_shape[reshape_dim] = output_size * interp_size;
      auto wts = empty(new_shape, CPU(c10::CppTypeToScalarType<scalar_t>()));
      auto strides = wts.strides().vec();
      strides[reshape_dim] = 0;
      new_shape[reshape_dim] = output_size;
      wts = wts.as_strided(new_shape, strides);
      output.emplace_back(wts);
      // Weights indices
      output.emplace_back(
          empty(new_shape, CPU(c10::CppTypeToScalarType<index_t>())));
    }

    int64_t* idx_ptr_xmin = output[0].data_ptr<index_t>();
    int64_t* idx_ptr_size = output[1].data_ptr<index_t>();
    int64_t* idx_ptr_stride = output[2].data_ptr<index_t>();
    scalar_t* wt_ptr = output[3].data_ptr<scalar_t>();
    int64_t* wt_idx_ptr = output[4].data_ptr<index_t>();

    int64_t xmin, xmax;

    for (int64_t i = 0; i < output_size; i++) {
      HelperInterpBase<index_t, scalar_t>::_compute_weights_aa(
          i,
          input_size,
          scale,
          support,
          wt_ptr + i * interp_size,
          interp_size,
          filter_fn,
          xmin,
          xmax);

      idx_ptr_xmin[i] = xmin * stride;
      idx_ptr_size[i] = xmax;
      idx_ptr_stride[i] = stride;
      wt_idx_ptr[i] = i * interp_size * sizeof(scalar_t);
    }
    return output;
  }
};

template <typename index_t, typename scalar_t>
struct HelperInterpLinear : public HelperInterpBase<index_t, scalar_t> {
  static const int interp_size = 2;

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L20-L29
  static inline scalar_t _filter(scalar_t x) {
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return 1.0 - x;
    }
    return 0.0;
  }

  static inline std::vector<Tensor> compute_indices_weights(
      int64_t input_size,
      int64_t output_size,
      int64_t stride,
      int64_t ndims,
      int64_t reshape_dim,
      bool align_corners,
      const c10::optional<double> opt_scale,
      bool antialias,
      int& out_interp_size) {
    TORCH_INTERNAL_ASSERT(antialias);
    scalar_t scale = area_pixel_compute_scale<scalar_t>(
        input_size, output_size, align_corners, opt_scale);

    out_interp_size = HelperInterpLinear<index_t, scalar_t>::interp_size;
    return HelperInterpLinear<index_t, scalar_t>::_compute_indices_weights_aa(
        input_size,
        output_size,
        stride,
        ndims,
        reshape_dim,
        align_corners,
        scale,
        out_interp_size,
        _filter);
  }
};

template <typename index_t, typename scalar_t>
struct HelperInterpCubic : public HelperInterpBase<index_t, scalar_t> {
  static const int interp_size = 4;

  static inline std::vector<Tensor> compute_indices_weights(
      int64_t input_size,
      int64_t output_size,
      int64_t stride,
      int64_t ndims,
      int64_t reshape_dim,
      bool align_corners,
      const c10::optional<double> opt_scale,
      bool antialias,
      int& out_interp_size) {
    TORCH_INTERNAL_ASSERT(antialias);
    scalar_t scale = area_pixel_compute_scale<scalar_t>(
        input_size, output_size, align_corners, opt_scale);

    out_interp_size = HelperInterpCubic<index_t, scalar_t>::interp_size;
    return HelperInterpCubic<index_t, scalar_t>::_compute_indices_weights_aa(
        input_size,
        output_size,
        stride,
        ndims,
        reshape_dim,
        align_corners,
        scale,
        out_interp_size,
        _filter);
  }

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L46-L62
  static inline scalar_t _filter(scalar_t x) {
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
#define a -0.5
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    }
    if (x < 2.0) {
      return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0.0;
#undef a
  }
};

template <
    typename index_t,
    int out_ndims,
    typename scale_type,
    template <typename, typename>
    class F>
void _ti_separable_upsample_generic_Nd_kernel_impl_single_dim(
    Tensor& output,
    const Tensor& input,
    int interp_dim,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {
  // input can be NCHW, NCL or NCKHW
  auto shape = input.sizes().vec();
  auto strides = input.strides().vec();
  auto oshape = output.sizes();

  TORCH_INTERNAL_ASSERT(
      shape.size() == oshape.size() && shape.size() == 2 + out_ndims);
  TORCH_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);
  TORCH_INTERNAL_ASSERT(antialias);

  for (int i = 0; i < out_ndims; i++) {
    shape[i + 2] = oshape[i + 2];
  }
  strides[interp_dim] = 0;
  auto restrided_input = input.as_strided(shape, strides);

  std::vector<std::vector<Tensor>> indices_weights;

  int interp_size = F<index_t, float>::interp_size;
  auto input_scalar_type = input.scalar_type();

  if (interp_size == 1 && input_scalar_type == at::ScalarType::Byte) {
    // nearest also supports uint8 tensor, but we have to use float
    // with compute_indices_weights
    input_scalar_type = at::ScalarType::Float;
  }

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Byte,
      input_scalar_type,
      "compute_indices_weights_generic",
      [&] {
        indices_weights.emplace_back(
            F<index_t, scalar_t>::compute_indices_weights(
                input.size(interp_dim),
                oshape[interp_dim],
                input.stride(interp_dim) * input.element_size(),
                input.dim(),
                interp_dim,
                align_corners,
                scales[interp_dim - 2],
                antialias,
                interp_size));
      });

  TensorIteratorConfig config;
  config.check_all_same_dtype(false)
      .declare_static_dtype_and_device(input.scalar_type(), input.device())
      .add_output(output)
      .add_input(restrided_input);

  for (auto& idx_weight : indices_weights) {
    for (auto& tensor : idx_weight) {
      config.add_input(tensor);
    }
  }

  auto iter = config.build();

  if (interp_size > 1) {
    // Nearest also supports uint8 tensor, so need to handle it separately
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "upsample_generic_Nd", [&] {
      ti_cpu_upsample_generic_aa<scalar_t, index_t, out_ndims>(
          iter, interp_size);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::Byte, iter.dtype(), "upsample_generic_Nd", [&] {
          ti_cpu_upsample_generic_aa<scalar_t, index_t, out_ndims>(
              iter, interp_size);
        });
  }
}

template <
    typename index_t,
    int out_ndims,
    typename scale_type,
    template <typename, typename>
    class F>
void ti_separable_upsample_generic_Nd_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    const scale_type& scales,
    bool antialias) {
  auto temp_oshape = input.sizes().vec();
  at::Tensor temp_output, temp_input = input;
  for (int i = 0; i < out_ndims - 1; i++) {
    int interp_dim = 2 + out_ndims - 1 - i;
    temp_oshape[interp_dim] = output.sizes()[interp_dim];
    temp_output = at::empty(temp_oshape, input.options());
    _ti_separable_upsample_generic_Nd_kernel_impl_single_dim<
        index_t,
        out_ndims,
        scale_t,
        F>(
        temp_output, temp_input, interp_dim, align_corners, scales, antialias);
    temp_input = temp_output;
  }
  _ti_separable_upsample_generic_Nd_kernel_impl_single_dim<
      index_t,
      out_ndims,
      scale_t,
      F>(output, temp_input, 2, align_corners, scales, antialias);
}

void _ti_upsample_bilinear2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool antialias) {
  ti_separable_upsample_generic_Nd_kernel_impl<
      int64_t,
      2,
      scale_t,
      HelperInterpLinear>(
      output, input, align_corners, {scales_h, scales_w}, antialias);
}

void _ti_upsample_bicubic2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool antialias) {
  ti_separable_upsample_generic_Nd_kernel_impl<
      int64_t,
      2,
      scale_t,
      HelperInterpCubic>(
      output, input, align_corners, {scales_h, scales_w}, antialias);
}

template <
    typename scalar_t,
    typename scale_type,
    template <typename, typename>
    class F>
void cpu_upsample_genNd_backward_aa(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(
      grad_input_.dtype() == grad_output_.dtype(),
      "expected dtype ",
      grad_output_.dtype(),
      " for `grad_input` but got dtype ",
      grad_input_.dtype());

  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();
  auto ndim = input_sizes.size();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  int64_t output_slice_size = output_depth * output_height * output_width;
  int interp_size = F<int64_t, float>::interp_size;

  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t c, int64_t h, int64_t w) {
      return grad_input_data + c * input_height * input_width +
          h * input_width + w;
    };

    const scalar_t support_h = (height_scale >= 1.0)
        ? (interp_size * 0.5) * height_scale
        : interp_size * 0.5;
    const scalar_t support_w = (width_scale >= 1.0)
        ? (interp_size * 0.5) * width_scale
        : interp_size * 0.5;

    const int interp_height = (int)ceilf(support_h) * 2 + 1;
    const int interp_width = (int)ceilf(support_w) * 2 + 1;

    std::vector<scalar_t> wx(interp_width, 0.0);
    std::vector<scalar_t> wy(interp_height, 0.0);

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t xmin, ymin;
    int64_t xsize, ysize;
    auto filter_fn = F<int64_t, scalar_t>::_filter;

    for (int64_t oh = 0; oh < output_height; oh++) {
      F<int64_t, scalar_t>::_compute_weights_aa(
          oh,
          input_height,
          height_scale,
          support_h,
          wy.data(),
          interp_height,
          filter_fn,
          ymin,
          ysize);

      for (int64_t ow = 0; ow < output_width; ow++) {
        F<int64_t, scalar_t>::_compute_weights_aa(
            ow,
            input_width,
            width_scale,
            support_w,
            wx.data(),
            interp_width,
            filter_fn,
            xmin,
            xsize);

        for (int64_t c = begin; c < end; c++) {
          scalar_t grad_output_value =
              grad_output_data[c * output_slice_size + oh * output_width + ow];

          for (size_t y = 0; y < ysize; y++) {
            for (size_t x = 0; x < xsize; x++) {
              *input_indexr(c, ymin + y, xmin + x) +=
                  wx[x] * wy[y] * grad_output_value;
            }
          }
        }
      }
    }
  };

  if (ndim == 4) {
    // upsample bilinear 2d
    at::parallel_for(
        0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    TORCH_CHECK(false, "Unsupported tensor ndim");
  }

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

void _upsample_bilinear2d_aa_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "upsample_bilinear2d_backward_cpu", [&] {
        cpu_upsample_genNd_backward_aa<scalar_t, scale_t, HelperInterpLinear>(
            grad_input, grad_output, align_corners, {scales_h, scales_w});
      });
}

void _upsample_bicubic2d_aa_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "upsample_bicubic2d_backward_cpu", [&] {
        cpu_upsample_genNd_backward_aa<scalar_t, scale_t, HelperInterpCubic>(
            grad_input, grad_output, align_corners, {scales_h, scales_w});
      });
}

} // namespace internal_upsample
} // namespace native
} // namespace at

namespace vision {
namespace ops {

namespace {

at::Tensor interpolate_bilinear2d_aa_forward_kernel(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners) {
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");

  c10::optional<c10::ArrayRef<double>> scale_factors = {};

  // Copied from UpSampleBilinear2d.cpp
  auto output = at::empty({0}, input.options());
  auto osize = at::native::upsample::compute_output_size(
      input.sizes(), output_size, scale_factors);
  auto scale_h = at::native::upsample::get_scale_value(scale_factors, 0);
  auto scale_w = at::native::upsample::get_scale_value(scale_factors, 1);
  auto full_output_size =
      at::native::upsample_2d_common_check(input.sizes(), osize);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  output.resize_(full_output_size, input.suggest_memory_format());
  at::native::internal_upsample::_ti_upsample_bilinear2d_kernel_impl(
      output, input, align_corners, scale_h, scale_w, /*antialias=*/true);
  return output;
}

at::Tensor interpolate_bicubic2d_aa_forward_kernel(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners) {
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");

  c10::optional<c10::ArrayRef<double>> scale_factors = {};

  // Copied from UpSampleBilinear2d.cpp
  auto output = at::empty({0}, input.options());
  auto osize = at::native::upsample::compute_output_size(
      input.sizes(), output_size, scale_factors);
  auto scale_h = at::native::upsample::get_scale_value(scale_factors, 0);
  auto scale_w = at::native::upsample::get_scale_value(scale_factors, 1);
  auto full_output_size =
      at::native::upsample_2d_common_check(input.sizes(), osize);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  output.resize_(full_output_size, input.suggest_memory_format());
  at::native::internal_upsample::_ti_upsample_bicubic2d_kernel_impl(
      output, input, align_corners, scale_h, scale_w, /*antialias=*/true);
  return output;
}

at::Tensor interpolate_bilinear2d_aa_backward_kernel(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners) {
  c10::optional<c10::ArrayRef<double>> scale_factors = {};

  // Copied from UpSampleBilinear2d.cpp::upsample_bilinear2d_backward
  auto grad_input = at::empty({0}, grad_output.options());
  auto osize = at::native::upsample::compute_output_size(
      input_size, output_size, scale_factors);
  auto scale_h = at::native::upsample::get_scale_value(scale_factors, 0);
  auto scale_w = at::native::upsample::get_scale_value(scale_factors, 1);

  auto full_output_size =
      at::native::upsample_2d_common_check(input_size, osize);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ",
      grad_output.dim());

  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_size[i],
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output.size(i));
  }

  grad_input.resize_(input_size, grad_output.suggest_memory_format());
  grad_input.zero_();
  at::native::internal_upsample::_upsample_bilinear2d_aa_backward_kernel_impl(
      grad_input, grad_output, align_corners, scale_h, scale_w);

  return grad_input;
}

at::Tensor interpolate_bicubic2d_aa_backward_kernel(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners) {
  c10::optional<c10::ArrayRef<double>> scale_factors = {};

  // Copied from UpSampleBicubic2d.cpp::upsample_bicubic2d_backward
  auto grad_input = at::empty({0}, grad_output.options());
  auto osize = at::native::upsample::compute_output_size(
      input_size, output_size, scale_factors);
  auto scale_h = at::native::upsample::get_scale_value(scale_factors, 0);
  auto scale_w = at::native::upsample::get_scale_value(scale_factors, 1);

  auto full_output_size =
      at::native::upsample_2d_common_check(input_size, osize);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ",
      grad_output.dim());

  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_size[i],
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output.size(i));
  }

  grad_input.resize_(input_size, grad_output.suggest_memory_format());
  grad_input.zero_();
  at::native::internal_upsample::_upsample_bicubic2d_aa_backward_kernel_impl(
      grad_input, grad_output, align_corners, scale_h, scale_w);

  return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_interpolate_bilinear2d_aa"),
      TORCH_FN(interpolate_bilinear2d_aa_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_interpolate_bicubic2d_aa"),
      TORCH_FN(interpolate_bicubic2d_aa_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_interpolate_bilinear2d_aa_backward"),
      TORCH_FN(interpolate_bilinear2d_aa_backward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_interpolate_bicubic2d_aa_backward"),
      TORCH_FN(interpolate_bicubic2d_aa_backward_kernel));
}

} // namespace ops
} // namespace vision
