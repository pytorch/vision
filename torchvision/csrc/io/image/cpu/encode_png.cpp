#include "encode_jpeg.h"

#include "common_png.h"

namespace vision {
namespace image {

#if !PNG_FOUND

torch::Tensor encode_png(const torch::Tensor& data, int64_t compression_level) {
  TORCH_CHECK(
      false, "encode_png: torchvision not compiled with libpng support");
}

#else

namespace {

struct torch_mem_encode {
  char* buffer;
  size_t size;
};

struct torch_png_error_mgr {
  const char* pngLastErrorMsg; /* error messages */
  jmp_buf setjmp_buffer; /* for return to caller */
};

using torch_png_error_mgr_ptr = torch_png_error_mgr*;

void torch_png_error(png_structp png_ptr, png_const_charp error_msg) {
  /* png_ptr->err really points to a torch_png_error_mgr struct, so coerce
   * pointer */
  auto error_ptr = (torch_png_error_mgr_ptr)png_get_error_ptr(png_ptr);
  /* Replace the error message on the error structure */
  error_ptr->pngLastErrorMsg = error_msg;
  /* Return control to the setjmp point */
  longjmp(error_ptr->setjmp_buffer, 1);
}

void torch_png_write_data(
    png_structp png_ptr,
    png_bytep data,
    png_size_t length) {
  struct torch_mem_encode* p =
      (struct torch_mem_encode*)png_get_io_ptr(png_ptr);
  size_t nsize = p->size + length;

  /* allocate or grow buffer */
  if (p->buffer)
    p->buffer = (char*)realloc(p->buffer, nsize);
  else
    p->buffer = (char*)malloc(nsize);

  if (!p->buffer)
    png_error(png_ptr, "Write Error");

  /* copy new bytes to end of buffer */
  memcpy(p->buffer + p->size, data, length);
  p->size += length;
}

} // namespace

torch::Tensor encode_png(const torch::Tensor& data, int64_t compression_level) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.io.image.cpu.encode_png.encode_png");
  // Define compression structures and error handling
  png_structp png_write;
  png_infop info_ptr;
  struct torch_png_error_mgr err_ptr;

  // Define output buffer
  struct torch_mem_encode buf_info;
  buf_info.buffer = NULL;
  buf_info.size = 0;

  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(err_ptr.setjmp_buffer)) {
    /* If we get here, the PNG code has signaled an error.
     * We need to clean up the PNG object and the buffer.
     */
    if (info_ptr != NULL) {
      png_destroy_info_struct(png_write, &info_ptr);
    }

    if (png_write != NULL) {
      png_destroy_write_struct(&png_write, NULL);
    }

    if (buf_info.buffer != NULL) {
      free(buf_info.buffer);
    }

    TORCH_CHECK(false, err_ptr.pngLastErrorMsg);
  }

  // Check that the compression level is between 0 and 9
  TORCH_CHECK(
      compression_level >= 0 && compression_level <= 9,
      "Compression level should be between 0 and 9");

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

  // Initialize PNG structures
  png_write = png_create_write_struct(
      PNG_LIBPNG_VER_STRING, &err_ptr, torch_png_error, NULL);

  info_ptr = png_create_info_struct(png_write);

  // Define custom buffer output
  png_set_write_fn(png_write, &buf_info, torch_png_write_data, NULL);

  // Set output image information
  auto color_type = channels == 1 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB;
  png_set_IHDR(
      png_write,
      info_ptr,
      width,
      height,
      8,
      color_type,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);

  // Set image compression level
  png_set_compression_level(png_write, compression_level);

  // Write file header
  png_write_info(png_write, info_ptr);

  auto stride = width * channels;
  auto ptr = input.data_ptr<uint8_t>();

  // Encode PNG file
  for (int y = 0; y < height; ++y) {
    png_write_row(png_write, ptr);
    ptr += stride;
  }

  // Write EOF
  png_write_end(png_write, info_ptr);

  // Destroy structures
  png_destroy_write_struct(&png_write, &info_ptr);

  torch::TensorOptions options = torch::TensorOptions{torch::kU8};
  auto outTensor = torch::empty({(long)buf_info.size}, options);

  // Copy memory from png buffer, since torch cannot get ownership of it via
  // `from_blob`
  auto outPtr = outTensor.data_ptr<uint8_t>();
  std::memcpy(outPtr, buf_info.buffer, sizeof(uint8_t) * outTensor.numel());
  free(buf_info.buffer);

  return outTensor;
}

#endif

} // namespace image
} // namespace vision
