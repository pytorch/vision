#include "encode_jpegs_cuda.h"
#if !NVJPEG_FOUND
namespace vision {
namespace image {

using namespace vision::stable;

std::vector<Tensor> encode_jpegs_cuda(
    const std::vector<Tensor>& decoded_images,
    const int64_t quality) {
  VISION_CHECK(
      false, "encode_jpegs_cuda: torchvision not compiled with nvJPEG support");
}
} // namespace image
} // namespace vision
#else

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace vision {
namespace image {

using namespace vision::stable;

// We use global variables to cache the encoder and decoder instances and
// reuse them across calls to the corresponding pytorch functions
std::mutex encoderMutex;
std::unique_ptr<CUDAJpegEncoder> cudaJpegEncoder;

std::vector<Tensor> encode_jpegs_cuda(
    const std::vector<Tensor>& decoded_images,
    const int64_t quality) {
  // Note: C10_LOG_API_USAGE_ONCE is not available in stable ABI

  // Some nvjpeg structures are not thread safe so we're keeping it single
  // threaded for now. In the future this may be an opportunity to unlock
  // further speedups
  std::lock_guard<std::mutex> lock(encoderMutex);
  VISION_CHECK(decoded_images.size() > 0, "Empty input tensor list");
  Device device = Device(
      decoded_images[0].device().type(), decoded_images[0].device().index());

  // Set the target CUDA device
  int prev_device;
  cudaGetDevice(&prev_device);
  int target_device_idx = device.has_index() ? device.index() : prev_device;
  cudaSetDevice(target_device_idx);

  // Create a device with the resolved index for consistency
  Device resolved_device(kCUDA, static_cast<int16_t>(target_device_idx));

  // lazy init of the encoder class
  // the encoder object holds on to a lot of state and is expensive to create,
  // so we reuse it across calls. NB: the cached structures are device specific
  // and cannot be reused across devices
  if (cudaJpegEncoder == nullptr ||
      resolved_device != cudaJpegEncoder->target_device) {
    if (cudaJpegEncoder != nullptr) {
      delete cudaJpegEncoder.release();
    }

    cudaJpegEncoder = std::make_unique<CUDAJpegEncoder>(resolved_device);

    // Unfortunately, we cannot rely on the smart pointer releasing the encoder
    // object correctly upon program exit. This is because, when cudaJpegEncoder
    // gets destroyed, the CUDA runtime may already be shut down, rendering all
    // destroy* calls in the encoder destructor invalid. Instead, we use an
    // atexit hook which executes after main() finishes, but hopefully before
    // CUDA shuts down when the program exits. If CUDA is already shut down the
    // destructor will detect this and will not attempt to destroy any encoder
    // structures.
    std::atexit([]() { delete cudaJpegEncoder.release(); });
  }

  std::vector<Tensor> contig_images;
  contig_images.reserve(decoded_images.size());
  for (const auto& image : decoded_images) {
    VISION_CHECK(
        image.scalar_type() == kByte, "Input tensor dtype should be uint8");

    VISION_CHECK(
        image.device().type() == device.type() &&
            image.device().index() == device.index(),
        "All input tensors must be on the same CUDA device when encoding with nvjpeg")

    VISION_CHECK(
        image.dim() == 3 && image.numel() > 0,
        "Input data should be a 3-dimensional tensor");

    VISION_CHECK(
        image.size(0) == 3,
        "The number of channels should be 3, got: ",
        image.size(0));

    // nvjpeg requires images to be contiguous
    if (is_contiguous(image)) {
      contig_images.push_back(image);
    } else {
      contig_images.push_back(torch::stable::contiguous(image));
    }
  }

  cudaJpegEncoder->set_quality(quality);
  std::vector<Tensor> encoded_images;
  for (const auto& image : contig_images) {
    auto encoded_image = cudaJpegEncoder->encode_jpeg(image);
    encoded_images.push_back(encoded_image);
  }

  // We use a dedicated stream to do the encoding and even though the results
  // may be ready on that stream we cannot assume that they are also available
  // on the current stream of the calling context when this function returns. We
  // use a blocking event to ensure that this is indeed the case. Crucially, we
  // do not want to block the host at this particular point
  // (which is what cudaStreamSynchronize would do.) Events allow us to
  // synchronize the streams without blocking the host.
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, cudaJpegEncoder->stream);
  cudaStreamWaitEvent(cudaJpegEncoder->current_stream, event, 0);
  cudaEventDestroy(event);

  // Restore original device
  cudaSetDevice(prev_device);
  return encoded_images;
}

CUDAJpegEncoder::CUDAJpegEncoder(const Device& target_device)
    : original_device{kCUDA, []() {
                        int dev;
                        cudaGetDevice(&dev);
                        return static_cast<int16_t>(dev);
                      }()},
      target_device{target_device},
      stream{nullptr},
      current_stream{nullptr} {
  // Create CUDA streams
  cudaStreamCreate(&stream);
  // Get the default stream (nullptr represents the default stream)
  current_stream = nullptr;

  nvjpegStatus_t status;
  status = nvjpegCreateSimple(&nvjpeg_handle);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg handle: ",
      status);

  status = nvjpegEncoderStateCreate(nvjpeg_handle, &nv_enc_state, stream);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg encoder state: ",
      status);

  status = nvjpegEncoderParamsCreate(nvjpeg_handle, &nv_enc_params, stream);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg encoder params: ",
      status);
}

CUDAJpegEncoder::~CUDAJpegEncoder() {
  /*
  The below code works on Mac and Linux, but fails on Windows.
  This is because on Windows, the atexit hook which calls this
  destructor executes after cuda is already shut down causing SIGSEGV.
  We do not have a solution to this problem at the moment, so we'll
  just leak the libnvjpeg & cuda variables for the time being and hope
  that the CUDA runtime handles cleanup for us.
  Please send a PR if you have a solution for this problem.
  */

  // if (stream != nullptr) {
  //   cudaStreamDestroy(stream);
  // }
}

Tensor CUDAJpegEncoder::encode_jpeg(const Tensor& src_image) {
  nvjpegStatus_t status;
  cudaError_t cudaStatus;

  // Ensure that the incoming src_image is safe to use
  cudaStatus = cudaStreamSynchronize(current_stream);
  VISION_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);

  int channels = src_image.size(0);
  int height = src_image.size(1);
  int width = src_image.size(2);

  status = nvjpegEncoderParamsSetSamplingFactors(
      nv_enc_params, NVJPEG_CSS_444, stream);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params sampling factors: ",
      status);

  nvjpegImage_t target_image;
  for (int c = 0; c < channels; c++) {
    target_image.channel[c] =
        torch::stable::select(src_image, 0, c).mutable_data_ptr<uint8_t>();
    // this is why we need contiguous tensors
    target_image.pitch[c] = width;
  }
  for (int c = channels; c < NVJPEG_MAX_COMPONENT; c++) {
    target_image.channel[c] = nullptr;
    target_image.pitch[c] = 0;
  }
  // Encode the image
  status = nvjpegEncodeImage(
      nvjpeg_handle,
      nv_enc_state,
      nv_enc_params,
      &target_image,
      NVJPEG_INPUT_RGB,
      width,
      height,
      stream);

  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS, "image encoding failed: ", status);
  // Retrieve length of the encoded image
  size_t length;
  status = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle, nv_enc_state, NULL, &length, stream);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image stream state: ",
      status);

  // Synchronize the stream to ensure that the encoded image is ready
  cudaStatus = cudaStreamSynchronize(stream);
  VISION_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);

  // Reserve buffer for the encoded image
  Tensor encoded_image =
      empty({static_cast<int64_t>(length)}, kByte, target_device);
  cudaStatus = cudaStreamSynchronize(stream);
  VISION_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);
  // Retrieve the encoded image
  status = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle,
      nv_enc_state,
      encoded_image.mutable_data_ptr<uint8_t>(),
      &length,
      stream);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image: ",
      status);
  return encoded_image;
}

void CUDAJpegEncoder::set_quality(const int64_t quality) {
  nvjpegStatus_t paramsQualityStatus =
      nvjpegEncoderParamsSetQuality(nv_enc_params, quality, stream);
  VISION_CHECK(
      paramsQualityStatus == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params quality: ",
      paramsQualityStatus);
}

} // namespace image
} // namespace vision

#endif // NVJPEG_FOUND
