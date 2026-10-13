#pragma once
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>
#include <tuple>
#include <vector>
#include "../common_stable.h"

#if NVJPEG_FOUND
#include <cuda_runtime.h>
#include <nvjpeg.h>

namespace vision {
namespace image {
class CUDAJpegDecoder {
 public:
  CUDAJpegDecoder(const torch::stable::Device& target_device);
  ~CUDAJpegDecoder();

  std::vector<torch::stable::Tensor> decode_images(
      const std::vector<torch::stable::Tensor>& encoded_images,
      const nvjpegOutputFormat_t& output_format);

  const torch::stable::Device original_device;
  const torch::stable::Device target_device;
  cudaStream_t stream;

 private:
  std::tuple<
      std::vector<nvjpegImage_t>,
      std::vector<torch::stable::Tensor>,
      std::vector<int>>
  prepare_buffers(
      const std::vector<torch::stable::Tensor>& encoded_images,
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

std::vector<torch::stable::Tensor> decode_jpegs_cuda(
    const std::vector<torch::stable::Tensor>& encoded_images,
    ImageReadMode mode,
    torch::stable::Device device);

} // namespace image
} // namespace vision
#endif
