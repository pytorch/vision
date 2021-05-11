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

namespace {
static nvjpegHandle_t nvjpeg_handle = nullptr;
}

torch::Tensor decode_jpeg_cuda(
    const torch::Tensor& data,
    ImageReadMode mode,
    torch::Device device) {
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");

  TORCH_CHECK(
      !data.is_cuda(),
      "The input tensor must be on CPU when decoding with nvjpeg")

  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  TORCH_CHECK(device.is_cuda(), "Expected a cuda device")

  at::cuda::CUDAGuard device_guard(device);

  // Create global nvJPEG handle
  std::once_flag nvjpeg_handle_creation_flag;
  std::call_once(nvjpeg_handle_creation_flag, []() {
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
  });

  // Create the jpeg state
  nvjpegJpegState_t jpeg_state;
  nvjpegStatus_t state_status =
      nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state);

  TORCH_CHECK(
      state_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegJpegStateCreate failed: ",
      state_status);

  auto datap = data.data_ptr<uint8_t>();

  // Get the image information
  int num_channels;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  nvjpegStatus_t info_status = nvjpegGetImageInfo(
      nvjpeg_handle,
      datap,
      data.numel(),
      &num_channels,
      &subsampling,
      widths,
      heights);

  if (info_status != NVJPEG_STATUS_SUCCESS) {
    nvjpegJpegStateDestroy(jpeg_state);
    TORCH_CHECK(false, "nvjpegGetImageInfo failed: ", info_status);
  }

  if (subsampling == NVJPEG_CSS_UNKNOWN) {
    nvjpegJpegStateDestroy(jpeg_state);
    TORCH_CHECK(false, "Unknown NVJPEG chroma subsampling");
  }

  int width = widths[0];
  int height = heights[0];

  nvjpegOutputFormat_t ouput_format;
  int num_channels_output;

  switch (mode) {
    case IMAGE_READ_MODE_UNCHANGED:
      num_channels_output = num_channels;
      // For some reason, setting output_format to NVJPEG_OUTPUT_UNCHANGED will
      // not properly decode RGB images (it's fine for grayscale), so we set
      // output_format manually here
      if (num_channels == 1) {
        ouput_format = NVJPEG_OUTPUT_Y;
      } else if (num_channels == 3) {
        ouput_format = NVJPEG_OUTPUT_RGB;
      } else {
        nvjpegJpegStateDestroy(jpeg_state);
        TORCH_CHECK(
            false,
            "When mode is UNCHANGED, only 1 or 3 input channels are allowed.");
      }
      break;
    case IMAGE_READ_MODE_GRAY:
      ouput_format = NVJPEG_OUTPUT_Y;
      num_channels_output = 1;
      break;
    case IMAGE_READ_MODE_RGB:
      ouput_format = NVJPEG_OUTPUT_RGB;
      num_channels_output = 3;
      break;
    default:
      nvjpegJpegStateDestroy(jpeg_state);
      TORCH_CHECK(
          false, "The provided mode is not supported for JPEG decoding on GPU");
  }

  auto out_tensor = torch::empty(
      {int64_t(num_channels_output), int64_t(height), int64_t(width)},
      torch::dtype(torch::kU8).device(device));

  // nvjpegImage_t is a struct with
  // - an array of pointers to each channel
  // - the pitch for each channel
  // which must be filled in manually
  nvjpegImage_t out_image;

  for (int c = 0; c < num_channels_output; c++) {
    out_image.channel[c] = out_tensor[c].data_ptr<uint8_t>();
    out_image.pitch[c] = width;
  }
  for (int c = num_channels_output; c < NVJPEG_MAX_COMPONENT; c++) {
    out_image.channel[c] = nullptr;
    out_image.pitch[c] = 0;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  nvjpegStatus_t decode_status = nvjpegDecode(
      nvjpeg_handle,
      jpeg_state,
      datap,
      data.numel(),
      ouput_format,
      &out_image,
      stream);

  nvjpegJpegStateDestroy(jpeg_state);

  TORCH_CHECK(
      decode_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegDecode failed: ",
      decode_status);

  return out_tensor;
}

#endif // NVJPEG_FOUND

} // namespace image
} // namespace vision
