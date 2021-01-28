#include "encode_jpeg.h"

#include "common_jpeg.h"

namespace vision {
namespace image {

#if !JPEG_FOUND

torch::Tensor encode_jpeg(const torch::Tensor& data, int64_t quality) {
  TORCH_CHECK(
      false, "encode_jpeg: torchvision not compiled with libjpeg support");
}

#else

using namespace detail;

torch::Tensor encode_jpeg(const torch::Tensor& data, int64_t quality) {
  // Define compression structures and error handling
  struct jpeg_compress_struct cinfo;
  struct torch_jpeg_error_mgr jerr;

  // Define buffer to write JPEG information to and its size
  unsigned long jpegSize = 0;
  uint8_t* jpegBuf = NULL;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = torch_jpeg_error_exit;

  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error.
     * We need to clean up the JPEG object and the buffer.
     */
    jpeg_destroy_compress(&cinfo);
    if (jpegBuf != NULL) {
      free(jpegBuf);
    }

    TORCH_CHECK(false, (const char*)jerr.jpegLastErrorMsg);
  }

  // Check that the input tensor is on CPU
  TORCH_CHECK(data.device() == torch::kCPU, "Input tensor should be on CPU");

  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Input tensor dtype should be uint8");

  // Check that the input tensor is 3-dimensional
  TORCH_CHECK(data.dim() == 3, "Input data should be a 3-dimensional tensor");

  // Get image info
  int channels = data.size(0);
  int height = data.size(1);
  int width = data.size(2);
  auto input = data.permute({1, 2, 0}).contiguous();

  TORCH_CHECK(
      channels == 1 || channels == 3,
      "The number of channels should be 1 or 3, got: ",
      channels);

  // Initialize JPEG structure
  jpeg_create_compress(&cinfo);

  // Set output image information
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = channels;
  cinfo.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  // Save JPEG output to a buffer
  jpeg_mem_dest(&cinfo, &jpegBuf, &jpegSize);

  // Start JPEG compression
  jpeg_start_compress(&cinfo, TRUE);

  auto stride = width * channels;
  auto ptr = input.data_ptr<uint8_t>();

  // Encode JPEG file
  while (cinfo.next_scanline < cinfo.image_height) {
    jpeg_write_scanlines(&cinfo, &ptr, 1);
    ptr += stride;
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  torch::TensorOptions options = torch::TensorOptions{torch::kU8};
  auto outTensor = torch::empty({(long)jpegSize}, options);

  // Copy memory from jpeg buffer, since torch cannot get ownership of it via
  // `from_blob`
  auto outPtr = outTensor.data_ptr<uint8_t>();
  std::memcpy(outPtr, jpegBuf, sizeof(uint8_t) * outTensor.numel());

  free(jpegBuf);

  return outTensor;
}
#endif

} // namespace image
} // namespace vision
