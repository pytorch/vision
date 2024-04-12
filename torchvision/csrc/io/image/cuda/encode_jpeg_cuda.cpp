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

#define YUV_IMG_SIZE_DIVISIBILITY 8

nvjpegHandle_t nvjpeg_handle = nullptr;
std::once_flag nvjpeg_handle_creation_flag;

#endif
#include <unistd.h>
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
    const nvjpegEncoderState_t nv_enc_state,
    const nvjpegEncoderParams_t nv_enc_params);

std::vector<torch::Tensor> encode_jpeg_cuda(
    const std::vector<torch::Tensor>& images,
    const int64_t quality) {
  C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cuda.encode_jpeg_cuda.encode_jpeg_cuda");

  TORCH_CHECK(images.size() > 0, "Empty input tensor list");
  for (const auto& image : images) {
    TORCH_CHECK(image.dtype() == torch::kU8, "Input tensor dtype should be uint8");

    TORCH_CHECK(
        image.is_cuda(),
        "All input tensors must be on a CUDA device when encoding with nvjpeg")

    // Due to the required input format of nvjpeg tensors must be contiguous in
    // memory. We could do the conversion to contiguous here but that comes with
    // a performance penalty which will be transparent to the user. Better to
    // make this explicit and push the conversion to user code.
    TORCH_CHECK(
        image.is_contiguous(),
        "All input tensors must be contiguous. Call tensor.contiguous() before calling this function.")

    TORCH_CHECK(
        image.dim() == 3 && image.numel() > 0,
        "Input data should be a 3-dimensional tensor");

    TORCH_CHECK(
        image.size(0) == 1 || image.size(0) == 3,
        "The number of channels should be 1 or 3, got: ", image.size(0));
  }

  torch::Device device = images[0].device();
  at::cuda::CUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // Create global nvJPEG handle
  std::call_once(::nvjpeg_handle_creation_flag, nvjpeg_init);

  nvjpegEncoderState_t nv_enc_state;
  nvjpegEncoderParams_t nv_enc_params;

  // initialize nvjpeg structures
  // these are rather expensive to create and thus will be reused across multiple calls to encode_single_jpeg
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
    for (const auto& image : images) {
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
    const nvjpegEncoderState_t nv_enc_state,
    const nvjpegEncoderParams_t nv_enc_params) {
  int channels = src_image.size(0);
  int height = src_image.size(1);
  int width = src_image.size(2);

  nvjpegStatus_t samplingSetResult = nvjpegEncoderParamsSetSamplingFactors(
      nv_enc_params, channels == 1 ? NVJPEG_CSS_GRAY : NVJPEG_CSS_444, stream);
  TORCH_CHECK(
      samplingSetResult == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params sampling factors: ",
      samplingSetResult);

  // Create nvjpeg image
  nvjpegImage_t target_image;

  // For some reason I couldn't get nvjpegEncodeImage to work for grayscale
  // images but nvjpegEncodeYUV seems to work fine. Annoyingly however,
  // nvjpegEncodeYUV requires the source image plane pitch to be divisible by 8
  // so we must pad the image if needed:
  // https://github.com/NVIDIA/cuda-samples/issues/23#issuecomment-559283013
  std::shared_ptr<torch::Tensor> padded_image_ptr =
      std::make_shared<torch::Tensor>(src_image);
  int padded_width = width;
  if (channels == 1 && width % YUV_IMG_SIZE_DIVISIBILITY != 0) {
    padded_width = ((width / YUV_IMG_SIZE_DIVISIBILITY) + 1) * YUV_IMG_SIZE_DIVISIBILITY;
    int left_padding, right_padding;
    left_padding = right_padding = (padded_width - width) / 2;
    if ((padded_width - width) % 2 != 0) {
      ++left_padding;
    }
    torch::nn::functional::PadFuncOptions pad_options(
        {left_padding, right_padding});
    auto padded_image =
        torch::nn::functional::pad(src_image, pad_options).contiguous();
    padded_image_ptr = std::make_shared<torch::Tensor>(padded_image);
  }

  for (int c = 0; c < channels; c++) {
    target_image.channel[c] = (*padded_image_ptr)[c].data_ptr<uint8_t>();
    // this is why we need contiguous tensors
    target_image.pitch[c] = padded_width;
  }
  for (int c = channels; c < NVJPEG_MAX_COMPONENT; c++) {
    target_image.channel[c] = nullptr;
    target_image.pitch[c] = 0;
  }
  nvjpegStatus_t encodingState;
  // Encode the image
  if (channels == 1) {
    // For some reason nvjpegEncodeImage fails for grayscale images, so we use
    // nvjpegEncodeYUV instead
    encodingState = nvjpegEncodeYUV(
        nvjpeg_handle,
        nv_enc_state,
        nv_enc_params,
        &target_image,
        NVJPEG_CSS_GRAY,
        padded_width,
        height,
        stream);
  } else {
    encodingState = nvjpegEncodeImage(
        nvjpeg_handle,
        nv_enc_state,
        nv_enc_params,
        &target_image,
        NVJPEG_INPUT_RGB,
        width,
        height,
        stream);
  }
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
          .device(torch::kCUDA, 1)
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
