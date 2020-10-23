#include "readpng_cpu.h"

// Comment
#include <ATen/ATen.h>
#include <setjmp.h>
#include <string>

#if !PNG_FOUND
torch::Tensor decodePNG(const torch::Tensor& data) {
  TORCH_CHECK(false, "decodePNG: torchvision not compiled with libPNG support");
}
#else
#include <png.h>

torch::Tensor decodePNG(const torch::Tensor& data) {
  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  auto png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  TORCH_CHECK(png_ptr, "libpng read structure allocation failed!")
  auto info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    // Seems redundant with the if statement. done here to avoid leaking memory.
    TORCH_CHECK(info_ptr, "libpng info structure allocation failed!")
  }

  auto datap = data.accessor<unsigned char, 1>().data();

  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    TORCH_CHECK(false, "Internal error.");
  }
  auto is_png = !png_sig_cmp(datap, 0, 8);
  TORCH_CHECK(is_png, "Content is not png!")

  struct Reader {
    png_const_bytep ptr;
  } reader;
  reader.ptr = png_const_bytep(datap) + 8;

  auto read_callback =
      [](png_structp png_ptr, png_bytep output, png_size_t bytes) {
        auto reader = static_cast<Reader*>(png_get_io_ptr(png_ptr));
        std::copy(reader->ptr, reader->ptr + bytes, output);
        reader->ptr += bytes;
      };
  png_set_sig_bytes(png_ptr, 8);
  png_set_read_fn(png_ptr, &reader, read_callback);
  png_read_info(png_ptr, info_ptr);

  png_uint_32 width, height;
  int bit_depth, color_type;
  auto retval = png_get_IHDR(
      png_ptr,
      info_ptr,
      &width,
      &height,
      &bit_depth,
      &color_type,
      nullptr,
      nullptr,
      nullptr);

  if (retval != 1) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    TORCH_CHECK(retval == 1, "Could read image metadata from content.")
  }
  if (color_type != PNG_COLOR_TYPE_RGB) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    TORCH_CHECK(
        color_type == PNG_COLOR_TYPE_RGB, "Non RGB images are not supported.")
  }

  auto tensor =
      torch::empty({int64_t(height), int64_t(width), int64_t(3)}, torch::kU8);
  auto ptr = tensor.accessor<uint8_t, 3>().data();
  auto bytes = png_get_rowbytes(png_ptr, info_ptr);
  for (decltype(height) i = 0; i < height; ++i) {
    png_read_row(png_ptr, ptr, nullptr);
    ptr += bytes;
  }
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  return tensor.permute({2, 0, 1});
}
#endif // PNG_FOUND
