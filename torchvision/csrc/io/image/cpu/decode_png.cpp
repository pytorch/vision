#include "decode_png.h"
#include "common_png.h"

namespace vision {
namespace image {

#if !PNG_FOUND
torch::Tensor decode_png(const torch::Tensor& data, ImageReadMode mode) {
  TORCH_CHECK(
      false, "decode_png: torchvision not compiled with libPNG support");
}
#else

torch::Tensor decode_png(const torch::Tensor& data, ImageReadMode mode) {
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
  int interlace_type;
  auto retval = png_get_IHDR(
      png_ptr,
      info_ptr,
      &width,
      &height,
      &bit_depth,
      &color_type,
      &interlace_type,
      nullptr,
      nullptr);

  if (retval != 1) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    TORCH_CHECK(retval == 1, "Could read image metadata from content.")
  }

  int channels = png_get_channels(png_ptr, info_ptr);

  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png_ptr);

  int number_of_passes;
  if (interlace_type == PNG_INTERLACE_ADAM7) {
    number_of_passes = png_set_interlace_handling(png_ptr);
  } else {
    number_of_passes = 1;
  }

  if (mode != IMAGE_READ_MODE_UNCHANGED) {
    // TODO: consider supporting PNG_INFO_tRNS
    bool is_palette = (color_type & PNG_COLOR_MASK_PALETTE) != 0;
    bool has_color = (color_type & PNG_COLOR_MASK_COLOR) != 0;
    bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0;

    switch (mode) {
      case IMAGE_READ_MODE_GRAY:
        if (color_type != PNG_COLOR_TYPE_GRAY) {
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
          channels = 1;
        }
        break;
      case IMAGE_READ_MODE_GRAY_ALPHA:
        if (color_type != PNG_COLOR_TYPE_GRAY_ALPHA) {
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
          channels = 2;
        }
        break;
      case IMAGE_READ_MODE_RGB:
        if (color_type != PNG_COLOR_TYPE_RGB) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
            has_alpha = true;
          } else if (!has_color) {
            png_set_gray_to_rgb(png_ptr);
          }

          if (has_alpha) {
            png_set_strip_alpha(png_ptr);
          }
          channels = 3;
        }
        break;
      case IMAGE_READ_MODE_RGB_ALPHA:
        if (color_type != PNG_COLOR_TYPE_RGB_ALPHA) {
          if (is_palette) {
            png_set_palette_to_rgb(png_ptr);
            has_alpha = true;
          } else if (!has_color) {
            png_set_gray_to_rgb(png_ptr);
          }

          if (!has_alpha) {
            png_set_add_alpha(png_ptr, (1 << bit_depth) - 1, PNG_FILLER_AFTER);
          }
          channels = 4;
        }
        break;
      default:
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        TORCH_CHECK(false, "The provided mode is not supported for PNG files");
    }

    png_read_update_info(png_ptr, info_ptr);
  }

  auto tensor = torch::empty(
      {int64_t(height), int64_t(width), channels},
      bit_depth <= 8 ? torch::kU8 : torch::kI32);

  if (bit_depth <= 8) {
    auto ptr = tensor.accessor<uint8_t, 3>().data();
    auto num_bytes_per_row = width * channels;
    for (int pass = 0; pass < number_of_passes; pass++) {
      for (png_uint_32 i = 0; i < height; ++i) {
        png_read_row(png_ptr, ptr, nullptr);
        ptr += num_bytes_per_row;
      }
      ptr = tensor.accessor<uint8_t, 3>().data();
    }
  } else {
    // We're reading a 16bits png, but pytorch doesn't support uint16. So we
    // read the whole image in an int32 tensor, and we'll convert it later to
    // a uint8
    auto ptr = (uint8_t*)tensor.accessor<int32_t, 3>().data();
    auto num_bytes_per_row = width * channels * 4;
    for (int pass = 0; pass < number_of_passes; pass++) {
      for (png_uint_32 i = 0; i < height; ++i) {
        png_read_row(png_ptr, ptr, nullptr);

        // We have a row that looks like this
        // [a, b, c, d, ., ., ., .]
        // where each letter is one byte - and so each consecutive pair of
        // letters corresponds to an actual 16bits pixel value. Half of the
        // cells on the right are empty because png_read_row() will read
        // num_pixels 16bits values, while our row has num_pixels 32bits cells
        // (and 32 = 2 * 16...)
        //
        // We need to re-organize the row for the int32 entries to be properly
        // interpreted with the same value as their 16bits counter-parts. On a
        // little-endian arch, we need a mapping like this:
        // [b, a, 0, 0, d, c, 0, 0]
        //  ^^^^        ^^^^
        // uint16      uint16
        //  ^^^^^^^^^^  ^^^^^^^^^^
        //     int32       int32
        // TODO: support big endian????
        for (int j = num_bytes_per_row - 1; j >= 0; j--) {
          switch (j % 4) {
            case 0:
              ptr[j] = ptr[j / 2 + 1];
              break;
            case 1:
              ptr[j] = ptr[j / 2];
              break;
            case 2:
              ptr[j] = 0;
              break;
            case 3:
              ptr[j] = 0;
              break;
            default:
              break;
          }
        }
        ptr += num_bytes_per_row;
      }
      ptr = (uint8_t*)tensor.accessor<int32_t, 3>().data();
    }
    // now let's pretend we had a uint8 tensor all along <o/
    tensor = (tensor / 65535. * 255.).to(torch::kU8);
  }
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  return tensor.permute({2, 0, 1});
}
#endif

} // namespace image
} // namespace vision
