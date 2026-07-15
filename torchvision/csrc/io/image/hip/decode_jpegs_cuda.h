#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>
#include <vector>

#include "../common_stable.h"

#if ROCJPEG_FOUND

#include <hip/hip_runtime.h>
#include <rocjpeg/rocjpeg.h>

// rocJPEG decode API documentation:
// https://rocm.docs.amd.com/projects/rocJPEG/en/latest/how-to/rocjpeg-decoding-a-jpeg-stream.html

namespace vision {
namespace image {
class RocJpegDecoder {
 public:
  RocJpegDecoder(const torch::stable::Device& target_device);
  ~RocJpegDecoder();

  std::vector<torch::stable::Tensor> decode_images(
      const std::vector<torch::stable::Tensor>& encoded_images,
      ImageReadMode mode);

  const torch::stable::Device target_device;

 private:
  std::vector<RocJpegStreamHandle> rocjpeg_stream_handles_;
  RocJpegHandle rocjpeg_handle_;
};

std::vector<torch::stable::Tensor> decode_jpegs_cuda(
    const std::vector<torch::stable::Tensor>& encoded_images,
    ImageReadMode mode,
    torch::stable::Device device);

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
