#include "readpng_cpu.h"

// Comment
#include <ATen/ATen.h>
#include <string>

#define PNG_FOUND 1
#if !PNG_FOUND
torch::Tensor decodePNG(const torch::Tensor& data, int64_t channels) {
  TORCH_CHECK(false, "decodePNG: torchvision not compiled with libPNG support");
}
#else
#include <png.h>
#include <setjmp.h>

torch::Tensor decodePNG(const torch::Tensor& data, int64_t channels) {
  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");
  TORCH_CHECK(
      channels >= 0 && channels <= 4, "Number of channels not supported");

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

  int current_channels = png_get_channels(png_ptr, info_ptr);

  if (channels > 0) {
    // TODO: consider supporting PNG_INFO_tRNS
    bool is_palette = (color_type & PNG_COLOR_MASK_PALETTE) != 0;
    bool has_color = (color_type & PNG_COLOR_MASK_COLOR) != 0;
    bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0;

    switch (channels) {
      case 1: // Gray
        if (is_palette) {
          png_set_palette_to_rgb(png_ptr);
          has_alpha = true;
        }

        if (has_alpha) {
          png_set_strip_alpha(png_ptr);
        }

        if (has_color) {
          png_set_rgb_to_gray(png_ptr, 1, 0.2989, 0.587);
        }
        break;
      case 2: // Gray + Alpha
        if (is_palette) {
          png_set_palette_to_rgb(png_ptr);
          has_alpha = true;
        }

        if (!has_alpha) {
          png_set_add_alpha(png_ptr, (1 << bit_depth) - 1, PNG_FILLER_AFTER);
        }

        if (has_color) {
          png_set_rgb_to_gray(png_ptr, 1, 0.2989, 0.587);
        }
        break;
      case 3:
        if (is_palette) {
          png_set_palette_to_rgb(png_ptr);
          has_alpha = true;
        } else if (!has_color) {
          png_set_gray_to_rgb(png_ptr);
        }

        if (has_alpha) {
          png_set_strip_alpha(png_ptr);
        }
        break;
      case 4:
        if (is_palette) {
          png_set_palette_to_rgb(png_ptr);
          has_alpha = true;
        } else if (!has_color) {
          png_set_gray_to_rgb(png_ptr);
        }

        if (!has_alpha) {
          png_set_add_alpha(png_ptr, (1 << bit_depth) - 1, PNG_FILLER_AFTER);
        }
        break;
      default:
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        TORCH_CHECK(false, "Invalid number of output channels.");
    }

    png_read_update_info(png_ptr, info_ptr);
  } else {
    channels = current_channels;
  }

  auto tensor =
      torch::empty({int64_t(height), int64_t(width), channels}, torch::kU8);
  auto ptr = tensor.accessor<uint8_t, 3>().data();
  auto bytes = png_get_rowbytes(png_ptr, info_ptr);
  for (png_uint_32 i = 0; i < height; ++i) {
    png_read_row(png_ptr, ptr, nullptr);
    ptr += bytes;
  }
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  return tensor.permute({2, 0, 1});
}
#endif // PNG_FOUND
