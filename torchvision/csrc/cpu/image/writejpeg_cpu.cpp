#include "writejpeg_cpu.h"

#include <setjmp.h>
#include <string>

#if !JPEG_FOUND

torch::Tensor encodeJPEG(const torch::Tensor& data) {
  TORCH_CHECK(
      false, "encodeJPEG: torchvision not compiled with libjpeg support");
}

void writeJPEG(const torch::Tensor& data, const char* filename) {
  TORCH_CHECK(
      false, "writeJPEG: torchvision not compiled with libjpeg support");
}

#else

#include <jpeglib.h>
#include "jpegcommon.h"

struct torch_jpeg_dst_mgr {
  struct jpeg_destination_mgr pub;
  uint8_t* out;
  JOCTET* buffer;
};

torch::Tensor encodeJPEG(const torch::Tensor& data, int quality) {
  // Define compression structures and error handling
  struct jpeg_compress_struct cinfo;
  struct torch_jpeg_error_mgr jerr;

  // Define buffer to write JPEG information to and its size
  u_long jpegSize = 0;
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

  // Get image info
  int channels, height, width;

  // Move tensor to CPU and cast it into uint8
  torch::Tensor input = data.to(torch::kCPU).to(torch::kU8);

  TORCH_CHECK(
      data.dim() >= 2 && data.dim() <= 3,
      "Input data should be a 3-dimensional or a 2-dimensional tensor");

  if (data.dim() == 3) {
    channels = data.size(0);
    height = data.size(1);
    width = data.size(2);
    input = input.permute({1, 2, 0});
  } else {
    channels = 1;
    height = data.size(1);
    width = data.size(2);
  }

  std::ostringstream channelErrS;
  channelErrS << "The number of channels should be 1 or 3, got: " << channels;

  const std::string& channelErrStr = channelErrS.str();
  const char* channelErr = channelErrStr.c_str();
  TORCH_CHECK(channels == 1 || channels == 3, channelErr);

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

void writeJPEG(const torch::Tensor& data, const char* filename, int quality) {
  auto jpegBuf = encodeJPEG(data, quality);
  auto fileBytes = jpegBuf.data_ptr<uint8_t>();
  FILE* outfile = fopen(filename, "wb");

  TORCH_CHECK(outfile != NULL, "Error opening output jpeg file");

  fwrite(fileBytes, sizeof(uint8_t), jpegBuf.numel(), outfile);
  fclose(outfile);
}

#endif
