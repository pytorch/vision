#pragma once
#include <vector>
#include "../../../StableABICompat.h"
#if NVJPEG_FOUND

#include <cuda_runtime.h>
#include <nvjpeg.h>

namespace vision {
namespace image {

class CUDAJpegEncoder {
 public:
  CUDAJpegEncoder(const vision::stable::Device& device);
  ~CUDAJpegEncoder();

  vision::stable::Tensor encode_jpeg(const vision::stable::Tensor& src_image);

  void set_quality(const int64_t quality);

  const vision::stable::Device original_device;
  const vision::stable::Device target_device;
  cudaStream_t stream;
  cudaStream_t current_stream;

 protected:
  nvjpegEncoderState_t nv_enc_state;
  nvjpegEncoderParams_t nv_enc_params;
  nvjpegHandle_t nvjpeg_handle;
};
} // namespace image
} // namespace vision
#endif
