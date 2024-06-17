#include "encode_jpegs_cuda.h"
#if !NVJPEG_FOUND
namespace vision {
namespace image {
std::vector<torch::Tensor> encode_jpegs_cuda(
    const std::vector<torch::Tensor>& decoded_images,
    const int64_t quality) {
  TORCH_CHECK(
      false, "encode_jpegs_cuda: torchvision not compiled with nvJPEG support");
}
} // namespace image
} // namespace vision
#else

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#include <torch/nn/functional.h>
#include <iostream>
#include <memory>
#include <string>
#include "c10/core/ScalarType.h"

namespace vision {
namespace image {

// We use global variables to cache the encoder and decoder instances and
// reuse them across calls to the corresponding pytorch functions
std::mutex encoderMutex;
std::unique_ptr<CUDAJpegEncoder> cudaJpegEncoder;

std::vector<torch::Tensor> encode_jpegs_cuda(
    const std::vector<torch::Tensor>& decoded_images,
    const int64_t quality) {
  C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cuda.encode_jpegs_cuda.encode_jpegs_cuda");

  // Some nvjpeg structures are not thread safe so we're keeping it single
  // threaded for now. In the future this may be an opportunity to unlock
  // further speedups
  std::lock_guard<std::mutex> lock(encoderMutex);
  TORCH_CHECK(decoded_images.size() > 0, "Empty input tensor list");
  torch::Device device = decoded_images[0].device();
  at::cuda::CUDAGuard device_guard(device);

  // lazy init of the encoder class
  // the encoder object holds on to a lot of state and is expensive to create,
  // so we reuse it across calls. NB: the cached structures are device specific
  // and cannot be reused across devices
  if (cudaJpegEncoder == nullptr || device != cudaJpegEncoder->target_device) {
    if (cudaJpegEncoder != nullptr)
      delete cudaJpegEncoder.release();

    cudaJpegEncoder = std::make_unique<CUDAJpegEncoder>(device);

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

  std::vector<torch::Tensor> contig_images;
  contig_images.reserve(decoded_images.size());
  for (const auto& image : decoded_images) {
    TORCH_CHECK(
        image.dtype() == torch::kU8, "Input tensor dtype should be uint8");

    TORCH_CHECK(
        image.device() == device,
        "All input tensors must be on the same CUDA device when encoding with nvjpeg")

    TORCH_CHECK(
        image.dim() == 3 && image.numel() > 0,
        "Input data should be a 3-dimensional tensor");

    TORCH_CHECK(
        image.size(0) == 3,
        "The number of channels should be 3, got: ",
        image.size(0));

    // nvjpeg requires images to be contiguous
    if (image.is_contiguous()) {
      contig_images.push_back(image);
    } else {
      contig_images.push_back(image.contiguous());
    }
  }

  cudaJpegEncoder->set_quality(quality);
  std::vector<torch::Tensor> encoded_images;
  at::cuda::CUDAEvent event;
  event.record(cudaJpegEncoder->stream);
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
  event.block(at::cuda::getCurrentCUDAStream(
      cudaJpegEncoder->original_device.has_index()
          ? cudaJpegEncoder->original_device.index()
          : 0));
  return encoded_images;
}

CUDAJpegEncoder::CUDAJpegEncoder(const torch::Device& target_device)
    : original_device{torch::kCUDA, torch::cuda::current_device()},
      target_device{target_device},
      stream{
          target_device.has_index()
              ? at::cuda::getStreamFromPool(false, target_device.index())
              : at::cuda::getStreamFromPool(false)} {
  nvjpegStatus_t status;
  status = nvjpegCreateSimple(&nvjpeg_handle);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg handle: ",
      status);

  status = nvjpegEncoderStateCreate(nvjpeg_handle, &nv_enc_state, stream);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg encoder state: ",
      status);

  status = nvjpegEncoderParamsCreate(nvjpeg_handle, &nv_enc_params, stream);
  TORCH_CHECK(
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

  // // We run cudaGetDeviceCount as a dummy to test if the CUDA runtime is
  // still
  // // initialized. If it is not, we can skip the rest of this function as it
  // is
  // // unsafe to execute.
  // int deviceCount = 0;
  // cudaError_t error = cudaGetDeviceCount(&deviceCount);
  // if (error != cudaSuccess)
  //   return; // CUDA runtime has already shut down. There's nothing we can do
  //           // now.

  // nvjpegStatus_t status;

  // status = nvjpegEncoderParamsDestroy(nv_enc_params);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg encoder params: ",
  //     status);

  // status = nvjpegEncoderStateDestroy(nv_enc_state);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg encoder state: ",
  //     status);

  // cudaStreamSynchronize(stream);

  // status = nvjpegDestroy(nvjpeg_handle);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS, "nvjpegDestroy failed: ", status);
}

torch::Tensor CUDAJpegEncoder::encode_jpeg(const torch::Tensor& src_image) {
  int channels = src_image.size(0);
  int height = src_image.size(1);
  int width = src_image.size(2);

  nvjpegStatus_t status;
  cudaError_t cudaStatus;
  status = nvjpegEncoderParamsSetSamplingFactors(
      nv_enc_params, NVJPEG_CSS_444, stream);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params sampling factors: ",
      status);

  nvjpegImage_t target_image;
  for (int c = 0; c < channels; c++) {
    target_image.channel[c] = src_image[c].data_ptr<uint8_t>();
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

  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS, "image encoding failed: ", status);
  // Retrieve length of the encoded image
  size_t length;
  status = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle, nv_enc_state, NULL, &length, stream);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image stream state: ",
      status);

  // Synchronize the stream to ensure that the encoded image is ready
  cudaStatus = cudaStreamSynchronize(stream);
  TORCH_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);

  // Reserve buffer for the encoded image
  torch::Tensor encoded_image = torch::empty(
      {static_cast<long>(length)},
      torch::TensorOptions()
          .dtype(torch::kByte)
          .layout(torch::kStrided)
          .device(target_device)
          .requires_grad(false));
  cudaStatus = cudaStreamSynchronize(stream);
  TORCH_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);
  // Retrieve the encoded image
  status = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle,
      nv_enc_state,
      encoded_image.data_ptr<uint8_t>(),
      &length,
      0);
  TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image: ",
      status);
  return encoded_image;
}

void CUDAJpegEncoder::set_quality(const int64_t quality) {
  nvjpegStatus_t paramsQualityStatus =
      nvjpegEncoderParamsSetQuality(nv_enc_params, quality, stream);
  TORCH_CHECK(
      paramsQualityStatus == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params quality: ",
      paramsQualityStatus);
}

} // namespace image
} // namespace vision

#endif // NVJPEG_FOUND
