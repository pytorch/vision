#include <torch/nn/functional.h>
#include "c10/core/ScalarType.h"
#include "encode_decode_jpeg_cuda.h"
#include "torch/types.h"

#include <ATen/ATen.h>
#include <iostream>
#include <memory>

#if NVJPEG_FOUND
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvjpeg.h>

nvjpegHandle_t nvjpeg_handle = nullptr;
std::once_flag nvjpeg_handle_creation_flag;

#endif
#include <string>

namespace vision {
namespace image {

#if !NVJPEG_FOUND

std::vector<torch::Tensor> encode_jpeg_cuda(
    const std::vector<torch::Tensor>& images,
    const int64_t quality) {
  TORCH_CHECK(
      false, "decode_jpeg_cuda: torchvision not compiled with nvJPEG support");
}

#else

void nvjpeg_init() {
  if (nvjpeg_handle == nullptr) {
    nvjpegStatus_t create_status = nvjpegCreateSimple(&nvjpeg_handle);

    if (create_status != NVJPEG_STATUS_SUCCESS) {
      // Reset handle so that one can still call the function again in the
      // same process if there was a failure
      free(nvjpeg_handle);
      nvjpeg_handle = nullptr;
    }
    TORCH_CHECK(
        create_status == NVJPEG_STATUS_SUCCESS,
        "nvjpegCreateSimple failed: ",
        create_status);
  }
}

torch::Tensor encode_single_jpeg(
    const torch::Tensor& data,
    const int64_t quality,
    const cudaStream_t stream,
    const torch::Device& device,
    const nvjpegEncoderState_t& nv_enc_state,
    const nvjpegEncoderParams_t& nv_enc_params);

std::vector<torch::Tensor> encode_jpeg_cuda(
    const std::vector<torch::Tensor>& images,
    const int64_t quality) {
  C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cuda.encode_jpeg_cuda.encode_jpeg_cuda");

  TORCH_CHECK(images.size() > 0, "Empty input tensor list");

  torch::Device device = images[0].device();
  at::cuda::CUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  std::vector<torch::Tensor> contig_images;
  contig_images.reserve(images.size());
  for (const auto& image : images) {
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
    contig_images.push_back(image.contiguous());
  }

  // Create global nvJPEG handle
  std::call_once(::nvjpeg_handle_creation_flag, nvjpeg_init);

  nvjpegEncoderState_t nv_enc_state;
  nvjpegEncoderParams_t nv_enc_params;

  // initialize nvjpeg structures
  // these are rather expensive to create and thus will be reused across
  // multiple calls to encode_single_jpeg
  try {
    nvjpegStatus_t stateCreateResult =
        nvjpegEncoderStateCreate(nvjpeg_handle, &nv_enc_state, stream);
    TORCH_CHECK(
        stateCreateResult == NVJPEG_STATUS_SUCCESS,
        "Failed to create nvjpeg encoder state: ",
        stateCreateResult);

    nvjpegStatus_t paramsCreateResult =
        nvjpegEncoderParamsCreate(nvjpeg_handle, &nv_enc_params, stream);
    TORCH_CHECK(
        paramsCreateResult == NVJPEG_STATUS_SUCCESS,
        "Failed to create nvjpeg encoder params: ",
        paramsCreateResult);

    nvjpegStatus_t paramsQualityStatus =
        nvjpegEncoderParamsSetQuality(nv_enc_params, quality, stream);
    TORCH_CHECK(
        paramsQualityStatus == NVJPEG_STATUS_SUCCESS,
        "Failed to set nvjpeg encoder params quality: ",
        paramsQualityStatus);

    std::vector<torch::Tensor> encoded_images;
    for (const auto& image : contig_images) {
      auto encoded_image = encode_single_jpeg(
          image, quality, stream, device, nv_enc_state, nv_enc_params);
      encoded_images.push_back(encoded_image);
    }
    // Clean up
    nvjpegEncoderStateDestroy(nv_enc_state);
    nvjpegEncoderParamsDestroy(nv_enc_params);
    return encoded_images;
  } catch (const std::exception& e) {
    nvjpegEncoderStateDestroy(nv_enc_state);
    nvjpegEncoderParamsDestroy(nv_enc_params);
    throw;
  }
}

torch::Tensor encode_single_jpeg(
    const torch::Tensor& src_image,
    const int64_t quality,
    const cudaStream_t stream,
    const torch::Device& device,
    const nvjpegEncoderState_t& nv_enc_state,
    const nvjpegEncoderParams_t& nv_enc_params) {
  int channels = src_image.size(0);
  int height = src_image.size(1);
  int width = src_image.size(2);

  nvjpegStatus_t samplingSetResult = nvjpegEncoderParamsSetSamplingFactors(
      nv_enc_params, NVJPEG_CSS_444, stream);
  TORCH_CHECK(
      samplingSetResult == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params sampling factors: ",
      samplingSetResult);

  // Create nvjpeg image
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
  nvjpegStatus_t encodingState;

  // Encode the image
  encodingState = nvjpegEncodeImage(
      nvjpeg_handle,
      nv_enc_state,
      nv_enc_params,
      &target_image,
      NVJPEG_INPUT_RGB,
      width,
      height,
      stream);

  TORCH_CHECK(
      encodingState == NVJPEG_STATUS_SUCCESS,
      "image encoding failed: ",
      encodingState);

  // Retrieve length of the encoded image
  size_t length;
  nvjpegStatus_t getStreamState = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle, nv_enc_state, NULL, &length, stream);
  TORCH_CHECK(
      getStreamState == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image stream state: ",
      getStreamState);

  // Synchronize the stream to ensure that the encoded image is ready
  cudaError_t syncState = cudaStreamSynchronize(stream);
  TORCH_CHECK(syncState == cudaSuccess, "CUDA ERROR: ", syncState);

  // Reserve buffer for the encoded image
  torch::Tensor encoded_image = torch::empty(
      {static_cast<long>(length)},
      torch::TensorOptions()
          .dtype(torch::kByte)
          .layout(torch::kStrided)
          .device(device)
          .requires_grad(false));
  syncState = cudaStreamSynchronize(stream);
  TORCH_CHECK(syncState == cudaSuccess, "CUDA ERROR: ", syncState);

  // Retrieve the encoded image
  getStreamState = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle,
      nv_enc_state,
      encoded_image.data_ptr<uint8_t>(),
      &length,
      0);
  TORCH_CHECK(
      getStreamState == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image: ",
      getStreamState);
  return encoded_image;
}

#endif // NVJPEG_FOUND

} // namespace image
} // namespace vision
