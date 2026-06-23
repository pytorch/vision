#pragma once

#if ROCJPEG_FOUND

#include <hip/hip_runtime.h>
#include <rocjpeg/rocjpeg.h>

#include <torch/types.h>
#include <vector>

#include "../common.h"

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
  std::vector<RocJpegStreamHandle> rocjpeg_stream_handles_;
  RocJpegHandle rocjpeg_handle_;
};
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
