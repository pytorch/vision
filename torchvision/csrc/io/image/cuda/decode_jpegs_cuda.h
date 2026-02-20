#pragma once
#include <vector>
#include "../../../StableABICompat.h"
#include "../common.h"

#if NVJPEG_FOUND
#include <cuda_runtime.h>
#include <nvjpeg.h>

namespace vision {
namespace image {
class CUDAJpegDecoder {
 public:
  CUDAJpegDecoder(const vision::stable::Device& target_device);
  ~CUDAJpegDecoder();

  std::vector<vision::stable::Tensor> decode_images(
      const std::vector<vision::stable::Tensor>& encoded_images,
      const nvjpegOutputFormat_t& output_format);

  const vision::stable::Device original_device;
  const vision::stable::Device target_device;
  cudaStream_t stream;

 private:
  std::tuple<
      std::vector<nvjpegImage_t>,
      std::vector<vision::stable::Tensor>,
      std::vector<int>>
  prepare_buffers(
      const std::vector<vision::stable::Tensor>& encoded_images,
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
