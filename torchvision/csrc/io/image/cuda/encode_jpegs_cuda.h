#pragma once
#include <vector>
#include "../../../StableABICompat.h"
#if NVJPEG_FOUND

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
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
  const c10::cuda::CUDAStream stream;
  const c10::cuda::CUDAStream current_stream;

 protected:
  nvjpegEncoderState_t nv_enc_state;
  nvjpegEncoderParams_t nv_enc_params;
  nvjpegHandle_t nvjpeg_handle;
};
} // namespace image
} // namespace vision
#endif
