#include <torch/library.h>
// Copied and adapted from
// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>

// Below is experimental temporary code before merging it to PyTorch
namespace at {
namespace native {
namespace internal_upsample {

__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t y,
    const size_t x) {
  return (nc * height + y) * width + x;
}

// taken from
// https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
// src/libImaging/Resample.c#L20-L29
template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t bilinear_filter(accscalar_t x) {
  if (x < 0.0) {
    x = -x;
  }
  if (x < 1.0) {
    return static_cast<accscalar_t>(1.0) - x;
  }
  return static_cast<accscalar_t>(0.0);
}

// taken from
// https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
// src/libImaging/Resample.c#L46-L62
template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t bicubic_filter(accscalar_t x) {
  // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
#define a -0.5
  if (x < 0.0) {
    x = -x;
  }
  if (x < 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + static_cast<accscalar_t>(1.0);
  }
  if (x < 2.0) {
    return (((x - 5) * x + 8) * x - 4) * a;
  }
  return static_cast<accscalar_t>(0.0);
#undef a
}

template <typename scalar_t, typename accscalar_t, typename filter_fn_t>
__device__ __forceinline__ static void _compute_weights(
    const int64_t i,
    const int64_t input_size,
    const accscalar_t scale,
    const accscalar_t support,
    scalar_t* wt_ptr,
    int64_t interp_size,
    filter_fn_t filter_fn,
    int64_t& xmin,
    int64_t& xmax) {
  accscalar_t invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  accscalar_t center = scale * (i + 0.5);
  xmin = max(
      static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
  xmax = min(static_cast<int64_t>(center + support + 0.5), input_size) - xmin;

  accscalar_t total_w = 0.0;
  int64_t j = 0;
  for (j = 0; j < xmax; j++) {
    accscalar_t w = filter_fn((j + xmin - center + 0.5) * invscale);
    wt_ptr[j] = static_cast<scalar_t>(w);
    total_w += w;
  }
  for (j = 0; j < xmax; j++) {
    if (total_w != 0.0) {
      wt_ptr[j] /= total_w;
    }
  }
  for (; j < interp_size; j++) {
    wt_ptr[j] = static_cast<scalar_t>(0.0);
  }
}

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static accscalar_t interpolate_aa_single_dim(
    scalar_t* src,
    scalar_t* weights,
    int64_t size) {
  scalar_t t = static_cast<accscalar_t>(*src);
  scalar_t wts = static_cast<accscalar_t>(weights[0]);
  accscalar_t output = t * wts;

  int64_t j = 1;
  for (; j < size; j++) {
    wts = static_cast<accscalar_t>(weights[j]);
    t = static_cast<accscalar_t>(*(src + j));
    output += t * wts;
  }
  return output;
}

template <typename scalar_t, typename accscalar_t, int interp_size>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_gen2d_out_frame(
    const int n,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor64<scalar_t, 4> idata,
    PackedTensorAccessor64<scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int height1 = idata.size(2);
  const int width1 = idata.size(3);
  const int height2 = odata.size(2);
  const int width2 = odata.size(3);

  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][h1][w1];
          odata[n][c][h2][w2] = val;
        }
      }
      return;
    }

    const accscalar_t support_h = static_cast<accscalar_t>(
        (rheight >= 1.0) ? (interp_size * 0.5) * rheight : interp_size * 0.5);
    const accscalar_t support_w = static_cast<accscalar_t>(
        (rwidth >= 1.0) ? (interp_size * 0.5) * rwidth : interp_size * 0.5);

    const int interp_height = (int)ceilf(support_h) * 2 + 1;
    const int interp_width = (int)ceilf(support_w) * 2 + 1;

    // Setup local buffers
    // TODO: maybe we can specify dynamic shared memory size before calling the
    // cuda code, however we should then ensure that device has enough shared
    // memory
    scalar_t wx[256];
    scalar_t wy[256];
    scalar_t buffer1[256];
    scalar_t buffer2[256];

    // Compute weights
    int64_t xmin, xsize, ymin, ysize;
    typedef scalar_t (*filter_fn_t)(scalar_t);
    if (interp_size == 2) {
      _compute_weights<scalar_t, accscalar_t, filter_fn_t>(
          w2,
          width1,
          rwidth,
          support_w,
          wx,
          interp_width,
          bilinear_filter,
          xmin,
          xsize);
      _compute_weights<scalar_t, accscalar_t, filter_fn_t>(
          h2,
          height1,
          rheight,
          support_h,
          wy,
          interp_height,
          bilinear_filter,
          ymin,
          ysize);
    } else if (interp_size == 4) {
      _compute_weights<scalar_t, accscalar_t, filter_fn_t>(
          w2,
          width1,
          rwidth,
          support_w,
          wx,
          interp_width,
          bicubic_filter,
          xmin,
          xsize);
      _compute_weights<scalar_t, accscalar_t, filter_fn_t>(
          h2,
          height1,
          rheight,
          support_h,
          wy,
          interp_height,
          bicubic_filter,
          ymin,
          ysize);
    }

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        // interpolate on x-axis for ymin to ymin + ysize
        for (int64_t y = 0; y < ysize; y++) {
          // copy data into the local buffer and use
          // interpolate_aa_single_dim method
          for (int x = 0; x < xsize; x++) {
            buffer1[x] = idata[n][c][ymin + y][xmin + x];
          }

          buffer2[y] = static_cast<scalar_t>(
              interpolate_aa_single_dim<scalar_t, accscalar_t>(
                  buffer1, wx, xsize));
        }
        odata[n][c][h2][w2] = static_cast<scalar_t>(
            interpolate_aa_single_dim<scalar_t, accscalar_t>(
                buffer2, wy, ysize));
      }
    }
  }
}

template <int interp_size>
static void upsample_gen2d_out_cuda_template(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_gen2d_out_cuda", {input_arg, output_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  const int num_kernels = output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor64<scalar_t, 4>();
        auto odata = output.packed_accessor64<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        // We are using static buffer memory of 256 * sizeof(float) per thread
        // to store weights. Size of weights array is
        // interp_size = scale * 2 + 1 for bilinear mode
        TORCH_CHECK(
            rheight < (255 / interp_size),
            "Max supported scale factor is 127 (bilinear), 63 (bicubic)");
        TORCH_CHECK(
            rwidth < (255 / interp_size),
            "Max supported scale factor is 127 (bilinear), 63 (bicubic)");

        upsample_gen2d_out_frame<scalar_t, accscalar_t, interp_size>
            <<<cuda::ATenCeilDiv(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels, rheight, rwidth, align_corners, idata, odata);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace internal_upsample
} // namespace native
} // namespace at

namespace vision {
namespace ops {

namespace {

// Copied from "UpSample.h" as we can not use UpSample.h with UpSample.cuh
static std::array<int64_t, 4> upsample_2d_common_check(
    at::IntArrayRef input_size,
    at::IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  return {nbatch, channels, output_height, output_width};
}

template <int interp_size>
at::Tensor interpolate_gen2d_aa_forward_kernel(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners) {
  c10::optional<c10::ArrayRef<double>> scale_factors = {};

  // Copied from UpSampleBilinear2d.cpp
  auto output = at::empty({0}, input.options());
  auto osize = at::native::upsample::compute_output_size(
      input.sizes(), output_size, scale_factors);
  auto scale_h = at::native::upsample_cuda::get_scale_value(scale_factors, 0);
  auto scale_w = at::native::upsample_cuda::get_scale_value(scale_factors, 1);

  auto full_output_size = upsample_2d_common_check(input.sizes(), osize);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  output.resize_(full_output_size, input.suggest_memory_format());

  at::native::internal_upsample::upsample_gen2d_out_cuda_template<interp_size>(
      output,
      input,
      {full_output_size[2], full_output_size[3]},
      align_corners,
      scale_h,
      scale_w);
  return output;
}

at::Tensor interpolate_linear_aa_forward_kernel(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners) {
  return interpolate_gen2d_aa_forward_kernel<2>(
      input, output_size, align_corners);
}

at::Tensor interpolate_bicubic_aa_forward_kernel(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners) {
  return interpolate_gen2d_aa_forward_kernel<4>(
      input, output_size, align_corners);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_interpolate_linear_aa"),
      TORCH_FN(interpolate_linear_aa_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_interpolate_bicubic_aa"),
      TORCH_FN(interpolate_bicubic_aa_forward_kernel));
}

} // namespace ops
} // namespace vision
