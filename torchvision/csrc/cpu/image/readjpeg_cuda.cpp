#include "readjpeg_cuda.h"

#include <ATen/ATen.h>
#include <string>

#if !NVJPEG_FOUND

torch::Tensor decodeJPEG_cuda(const torch::Tensor& data) {
  TORCH_CHECK(
      false, "decodeJPEG_cuda: torchvision not compiled with nvJPEG support");
}

#else

#include <nvjpeg.h>

void init_nvjpegImage(nvjpegImage_t& img) {
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    img.channel[c] = NULL;
    img.pitch[c] = 0;
  }
}

torch::Tensor decodeJPEG_cuda(const torch::Tensor& data) {
  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  auto datap = data.data_ptr<uint8_t>();

  // Create nvJPEG handle (TODO: only initialise the library once)
  nvjpegHandle_t nvjpeg_handle;
  nvjpegStatus_t create_status = nvjpegCreateSimple(&nvjpeg_handle);

  TORCH_CHECK(
      create_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegCreateSimple failed: ",
      create_status);

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

  TORCH_CHECK(
      info_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegGetImageInfo failed: ",
      info_status);

  TORCH_CHECK(components == 3, "Only RGB for now");

  int width = widths[0];
  int height = heights[0];

  // nvjpegImage_t is a struct with
  // - an array of pointers to each channel
  // - the pitch for each channel
  // which must be filled in manually
  nvjpegImage_t outImage;
  init_nvjpegImage(outImage);

  // TODO device selection
  auto tensor = torch::empty(
      {int64_t(components), int64_t(height), int64_t(width)},
      torch::dtype(torch::kU8).device(torch::kCUDA));

  for (int c = 0; c < 3; c++) {
    outImage.channel[c] = tensor[c].data_ptr<uint8_t>();
    outImage.pitch[c] = width;
  }

  // TODO torch cuda stream support
  // TODO output besides RGB
  nvjpegStatus_t decode_status = nvjpegDecode(
      nvjpeg_handle,
      nvjpeg_state,
      datap,
      data.numel(),
      NVJPEG_OUTPUT_RGB,
      &outImage,
      /*stream=*/0);

  TORCH_CHECK(
      decode_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegDecode failed: ",
      decode_status);

  // Destroy the state and (for now) library handle
  nvjpegJpegStateDestroy(nvjpeg_state);
  nvjpegDestroy(nvjpeg_handle);

  return tensor;
}

#endif // NVJPEG_FOUND