#include "decode_jpeg_cuda.h"

#include <ATen/ATen.h>

#if NVJPEG_FOUND
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvjpeg.h>
#endif

#include <string>

namespace vision {
namespace image {

#if !NVJPEG_FOUND

torch::Tensor decode_jpeg_cuda(
    const torch::Tensor& data,
    ImageReadMode mode,
    torch::Device device) {
  TORCH_CHECK(
      false, "decode_jpeg_cuda: torchvision not compiled with nvJPEG support");
}

#else

static nvjpegHandle_t nvjpeg_handle = nullptr;

void init_nvjpegImage(nvjpegImage_t& img) {
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    img.channel[c] = nullptr;
    img.pitch[c] = 0;
  }
}

torch::Tensor decode_jpeg_cuda(
    const torch::Tensor& data,
    ImageReadMode mode,
    torch::Device device) {
  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  TORCH_CHECK(
    device.is_cuda(), "Expected a cuda device"
  )

  at::cuda::CUDAGuard device_guard(device);

  auto datap = data.data_ptr<uint8_t>();

  // Create nvJPEG handle
  if (nvjpeg_handle == nullptr) {
    nvjpegStatus_t create_status = nvjpegCreateSimple(&nvjpeg_handle);

    TORCH_CHECK(
        create_status == NVJPEG_STATUS_SUCCESS,
        "nvjpegCreateSimple failed: ",
        create_status);
  }

  // Create nvJPEG state (should this be persistent or not?)
  nvjpegJpegState_t nvjpeg_state;
  nvjpegStatus_t state_status =
      nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);

  TORCH_CHECK(
      state_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegJpegStateCreate failed: ",
      state_status);

  // Get the image information
  int components;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  nvjpegStatus_t info_status = nvjpegGetImageInfo(
      nvjpeg_handle,
      datap,
      data.numel(),
      &components,
      &subsampling,
      widths,
      heights);

  if (info_status != NVJPEG_STATUS_SUCCESS) {
    nvjpegJpegStateDestroy(nvjpeg_state);
    TORCH_CHECK(false, "nvjpegGetImageInfo failed: ", info_status);
  }

  if (subsampling == NVJPEG_CSS_UNKNOWN) {
    nvjpegJpegStateDestroy(nvjpeg_state);
    TORCH_CHECK(false, "Unknown NVJPEG chroma subsampling");
  }

  int width = widths[0];
  int height = heights[0];

  nvjpegOutputFormat_t outputFormat;
  int outputComponents;

  switch (mode) {
    case IMAGE_READ_MODE_UNCHANGED:
      if (components == 1) {
        outputFormat = NVJPEG_OUTPUT_Y;
        outputComponents = 1;
      } else if (components == 3) {
        outputFormat = NVJPEG_OUTPUT_RGB;
        outputComponents = 3;
      } else {
        nvjpegJpegStateDestroy(nvjpeg_state);
        TORCH_CHECK(
            false, "The provided mode is not supported for JPEG files on GPU");
      }
      break;
    case IMAGE_READ_MODE_GRAY:
      // This will do 0.299*R + 0.587*G + 0.114*B like opencv
      // TODO check if that is the same as libjpeg
      outputFormat = NVJPEG_OUTPUT_Y;
      outputComponents = 1;
      break;
    case IMAGE_READ_MODE_RGB:
      outputFormat = NVJPEG_OUTPUT_RGB;
      outputComponents = 3;
      break;
    default:
      // CMYK as input might work with nvjpegDecodeParamsSetAllowCMYK()
      nvjpegJpegStateDestroy(nvjpeg_state);
      TORCH_CHECK(
          false, "The provided mode is not supported for JPEG files on GPU");
  }

  // nvjpegImage_t is a struct with
  // - an array of pointers to each channel
  // - the pitch for each channel
  // which must be filled in manually
  nvjpegImage_t outImage;
  init_nvjpegImage(outImage);

  // TODO device selection
  auto tensor = torch::empty(
      {int64_t(outputComponents), int64_t(height), int64_t(width)},
      torch::dtype(torch::kU8).device(device));

  for (int c = 0; c < outputComponents; c++) {
    outImage.channel[c] = tensor[c].data_ptr<uint8_t>();
    outImage.pitch[c] = width;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  nvjpegStatus_t decode_status = nvjpegDecode(
      nvjpeg_handle,
      nvjpeg_state,
      datap,
      data.numel(),
      outputFormat,
      &outImage,
      stream);

  // Destroy the state
  nvjpegJpegStateDestroy(nvjpeg_state);

  TORCH_CHECK(
      decode_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegDecode failed: ",
      decode_status);

  return tensor;
}


torch::Tensor decode_jpeg_batch_cuda(
    const torch::Tensor& data,
    ImageReadMode mode,
    torch::Device device,
    int64_t batch_size,
    int64_t height,
    int64_t width) {
  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  TORCH_CHECK(
    device.is_cuda(), "Expected a cuda device"
  )

  at::cuda::CUDAGuard device_guard(device);

  // Create nvJPEG handle
  if (nvjpeg_handle == nullptr) {
    nvjpegStatus_t create_status = nvjpegCreateSimple(&nvjpeg_handle);

    TORCH_CHECK(
        create_status == NVJPEG_STATUS_SUCCESS,
        "nvjpegCreateSimple failed: ",
        create_status);
  }

  // Create nvJPEG state (should this be persistent or not?)
  nvjpegJpegState_t nvjpeg_state;
  nvjpegStatus_t state_status =
      nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);

  TORCH_CHECK(
      state_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegJpegStateCreate failed: ",
      state_status);

  nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_RGB;
  int outputComponents = 3;
  int max_cpu_threads = 1;

  nvjpegStatus_t decode_batched_initialize_state =
      nvjpegDecodeBatchedInitialize(
          nvjpeg_handle, nvjpeg_state, batch_size, max_cpu_threads, output_format);

  TORCH_CHECK(
      decode_batched_initialize_state == NVJPEG_STATUS_SUCCESS,
      "nvjpegDecodeBatchedInitialize failed: ",
      decode_batched_initialize_state);

  std::vector<nvjpegImage_t> iout(batch_size);
  std::vector<size_t> lengths(batch_size);

  auto datap = data.data_ptr<uint8_t>();
  std::vector<const unsigned char*> batched_bitstreams;
  int single_encoded_jpeg_size = data.numel() / batch_size;  // TODO change this. This assume all the images in the batch are the same.

  auto tensor = torch::empty(
      {batch_size, int64_t(outputComponents), int64_t(height), int64_t(width)},
      torch::dtype(torch::kU8).device(device));

  for (size_t img_idx = 0; img_idx < batch_size; img_idx++) {
    init_nvjpegImage(iout[img_idx]);
    lengths[img_idx] = single_encoded_jpeg_size;
    batched_bitstreams.push_back((const unsigned char*)(datap + (img_idx * single_encoded_jpeg_size)));
    for (int c = 0; c < outputComponents; c++) {
      iout[img_idx].channel[c] = tensor[img_idx][c].data_ptr<uint8_t>();
      iout[img_idx].pitch[c] = width;
    }
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  nvjpegStatus_t decode_batched_status = nvjpegDecodeBatched(
      nvjpeg_handle,
      nvjpeg_state,
      batched_bitstreams.data(),
      lengths.data(),
      iout.data(),
      stream
  );

  // Destroy the state
  nvjpegJpegStateDestroy(nvjpeg_state);

  TORCH_CHECK(
      decode_batched_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegDecodeBatched failed: ",
      decode_batched_status);

  return tensor;
}

#endif // NVJPEG_FOUND

} // namespace image
} // namespace vision
