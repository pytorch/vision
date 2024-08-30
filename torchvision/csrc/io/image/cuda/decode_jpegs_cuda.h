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
