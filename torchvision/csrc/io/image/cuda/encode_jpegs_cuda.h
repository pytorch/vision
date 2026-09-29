#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>
#include <vector>

#if NVJPEG_FOUND

#include <cuda_runtime.h>
#include <nvjpeg.h>

namespace vision {
namespace image {

class CUDAJpegEncoder {
 public:
  CUDAJpegEncoder(const torch::stable::Device& target_device);
  ~CUDAJpegEncoder();

  torch::stable::Tensor encode_jpeg(const torch::stable::Tensor& src_image);

  void set_quality(const int64_t quality);

  const torch::stable::Device original_device;
  const torch::stable::Device target_device;
  cudaStream_t stream;
  cudaStream_t current_stream;

 protected:
  nvjpegEncoderState_t nv_enc_state;
  nvjpegEncoderParams_t nv_enc_params;
  nvjpegHandle_t nvjpeg_handle;
};

std::vector<torch::stable::Tensor> encode_jpegs_cuda(
    const std::vector<torch::stable::Tensor>& decoded_images,
    const int64_t quality);

} // namespace image
} // namespace vision
#endif
