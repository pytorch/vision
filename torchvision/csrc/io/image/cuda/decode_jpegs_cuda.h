#pragma once
#include <torch/types.h>
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
      const nvjpegOutputFormat_t& output_format);

  const torch::Device original_device;
  const torch::Device target_device;
  const c10::cuda::CUDAStream stream;

 private:
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
} // namespace image
} // namespace vision

#endif

#if ROCJPEG_FOUND

#include <c10/cuda/CUDAStream.h>
#include <rocjpeg/rocjpeg.h>

namespace vision {
namespace image {
class RocJpegDecoder {
 public:
  RocJpegDecoder(const torch::Device& target_device);
  ~RocJpegDecoder();

  std::vector<torch::Tensor> decode_images(
      const std::vector<torch::Tensor>& encoded_images,
      const RocJpegOutputFormat& output_format);

  const torch::Device original_device;
  const torch::Device target_device;
  const c10::cuda::CUDAStream stream;

 private:
  RocJpegStreamHandle rocjpeg_stream_handles[2];
  RocJpegHandle rocjpeg_handle;
};
} // namespace image
} // namespace vision

#define CHECK_ROCJPEG(call)                                                  \
  {                                                                          \
    RocJpegStatus rocjpeg_status = (call);                                   \
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                          \
      std::cerr << #call << " returned "                                     \
                << rocJpegGetErrorName(rocjpeg_status) << " at " << __FILE__ \
                << ":" << __LINE__ << std::endl;                             \
      exit(1);                                                               \
    }                                                                        \
  }

#define CHECK_HIP(call)                                                    \
  {                                                                        \
    hipError_t hip_status = (call);                                        \
    if (hip_status != hipSuccess) {                                        \
      std::cout << "HIP failure: 'status: " << hipGetErrorName(hip_status) \
                << "' at " << __FILE__ << ":" << __LINE__ << std::endl;    \
      exit(1);                                                             \
    }                                                                      \
  }

#endif
