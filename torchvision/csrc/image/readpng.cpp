#include "readpng.h"

#include <sxx/sxx.h>

#include <png.h>

namespace torch {
namespace vision {
namespace image {
namespace impl {
bool is_png(const void* data) {
  return png_sig_cmp(png_const_bytep(data), 0, 8) == 0;
}

torch::Tensor read_png(const void* data) {
  struct Reader {
    png_const_bytep ptr;
  } reader;

  reader.ptr = png_const_bytep(data) + 8;
  auto read_callback =
      [](png_structp png_ptr, png_bytep output, png_size_t bytes) {
        auto reader = static_cast<Reader*>(png_get_io_ptr(png_ptr));
        std::copy(reader->ptr, reader->ptr + bytes, output);
        reader->ptr += bytes;
      };

  auto png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr)
    return torch::empty({});

  auto info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    return torch::empty({});
  }

  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    return torch::empty({});
  }

  png_set_read_fn(png_ptr, &reader, read_callback);
  png_set_sig_bytes(png_ptr, 8);
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

  if (retval != 1 || color_type != PNG_COLOR_TYPE_RGB) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    return torch::empty({});
  }

  auto tensor =
      torch::empty({int64_t(height), int64_t(width), int64_t(3)}, torch::kU8);
  auto ptr = tensor.data<uint8_t>();
  auto bytes = png_get_rowbytes(png_ptr, info_ptr);

  for (decltype(height) i = 0; i < height; ++i) {
    png_read_row(png_ptr, ptr, nullptr);
    ptr += bytes;
  }

  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  return tensor;
}

} // namespace impl
} // namespace image
} // namespace vision
} // namespace torch
