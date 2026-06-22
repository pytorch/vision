#pragma once
#include <torch/types.h>
#include <cstddef>
#include <vector>
#include "../common.h"

#if NVJPEG_FOUND

#include <c10/cuda/CUDAStream.h>
#include <nvjpeg.h>

namespace vision {
namespace image {
class CUDAJpegDecoder {
 public:
  CUDAJpegDecoder(const torch::Device& target_device);
  ~CUDAJpegDecoder();

  std::vector<torch::Tensor> decode_images(
      const std::vector<torch::Tensor>& encoded_images,
      vision::image::ImageReadMode mode);

  const torch::Device original_device;
  const torch::Device target_device;
  const c10::cuda::CUDAStream stream;

 private:
  std::vector<torch::Tensor> decode_images(
      const std::vector<torch::Tensor>& encoded_images,
      const nvjpegOutputFormat_t& output_format);
  std::tuple<
      std::vector<nvjpegImage_t>,
      std::vector<torch::Tensor>,
      std::vector<int>>
  prepare_buffers(
      const std::vector<torch::Tensor>& encoded_images,
      const nvjpegOutputFormat_t& output_format);
  nvjpegJpegState_t nvjpeg_state;
  nvjpegJpegState_t nvjpeg_decoupled_state;
  nvjpegBufferPinned_t pinned_buffers[2];
  nvjpegBufferDevice_t device_buffer;
  nvjpegJpegStream_t jpeg_streams[2];
  nvjpegDecodeParams_t nvjpeg_decode_params;
  nvjpegJpegDecoder_t nvjpeg_decoder;
  bool hw_decode_available{false};
  nvjpegHandle_t nvjpeg_handle;
};
using GpuJpegDecoder = CUDAJpegDecoder;
} // namespace image
} // namespace vision

#elif ROCJPEG_FOUND

#include <hip/hip_runtime.h>
#include <rocjpeg/rocjpeg.h>

// rocJPEG decode API documentation:
// https://rocm.docs.amd.com/projects/rocJPEG/en/latest/how-to/rocjpeg-decoding-a-jpeg-stream.html

namespace vision {
namespace image {
class RocJpegDecoder {
 public:
  RocJpegDecoder(const torch::Device& target_device);
  ~RocJpegDecoder();

  std::vector<torch::Tensor> decode_images(
      const std::vector<torch::Tensor>& encoded_images,
      vision::image::ImageReadMode mode);

  const torch::Device target_device;

 private:
  void ensure_stream_handle_count(std::size_t num_handles);

  std::vector<RocJpegStreamHandle> rocjpeg_stream_handles_;
  RocJpegHandle rocjpeg_handle_;
};
using GpuJpegDecoder = RocJpegDecoder;
} // namespace image
} // namespace vision

#define CHECK_ROCJPEG(call)                       \
  {                                               \
    RocJpegStatus rocjpeg_status = (call);        \
    STD_TORCH_CHECK(                              \
        rocjpeg_status == ROCJPEG_STATUS_SUCCESS, \
        #call,                                    \
        " returned ",                             \
        rocJpegGetErrorName(rocjpeg_status));     \
  }

#define CHECK_HIP(call)               \
  {                                   \
    hipError_t hip_status = (call);   \
    STD_TORCH_CHECK(                  \
        hip_status == hipSuccess,     \
        #call,                        \
        " failed with status: ",      \
        hipGetErrorName(hip_status)); \
  }

#endif
