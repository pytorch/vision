#include "decode_png.h"
#include "../common.h"
#include "common_png.h"
#include "exif.h"

namespace vision {
namespace image {

using namespace exif_private;

#if !PNG_FOUND
torch::Tensor decode_png(
    const torch::Tensor& data,
    ImageReadMode mode,
    bool apply_exif_orientation) {
  TORCH_CHECK(
      false, "decode_png: torchvision not compiled with libPNG support");
}
#else

bool is_little_endian() {
  uint32_t x = 1;
  return *(uint8_t*)&x;
}

torch::Tensor decode_png(
    const torch::Tensor& data,
    ImageReadMode mode,
    bool apply_exif_orientation) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.io.image.cpu.decode_png.decode_png");

  validate_encoded_data(data);

  auto png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  TORCH_CHECK(png_ptr, "libpng read structure allocation failed!")
  auto info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    // Seems redundant with the if statement. done here to avoid leaking memory.
    TORCH_CHECK(info_ptr, "libpng info structure allocation failed!")
  }

  auto accessor = data.accessor<unsigned char, 1>();
  auto datap = accessor.data();
  auto datap_len = accessor.size(0);

  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    TORCH_CHECK(false, "Internal error.");
  }
  TORCH_CHECK(datap_len >= 8, "Content is too small for png!")
  auto is_png = !png_sig_cmp(datap, 0, 8);
  TORCH_CHECK(is_png, "Content is not png!")

  struct Reader {
    png_const_bytep ptr;
    png_size_t count;
  } reader;
  reader.ptr = png_const_bytep(datap) + 8;
  reader.count = datap_len - 8;

  auto read_callback = [](png_structp png_ptr,
                          png_bytep output,
                          png_size_t bytes) {
    auto reader = static_cast<Reader*>(png_get_io_ptr(png_ptr));
    TORCH_CHECK(
        reader->count >= bytes,
        "Out of bound read in decode_png. Probably, the input image is corrupted");
    std::copy(reader->ptr, reader->ptr + bytes, output);
    reader->ptr += bytes;
    reader->count -= bytes;
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

  if (bit_depth > 8 && bit_depth != 16) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    TORCH_CHECK(
        false,
        "bit depth of png image is " + std::to_string(bit_depth) +
            ". Only <=8 and 16 are supported.")
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

  auto num_pixels_per_row = width * channels;
  auto is_16_bits = bit_depth == 16;
  auto tensor = torch::empty(
      {int64_t(height), int64_t(width), channels},
      is_16_bits ? at::kUInt16 : torch::kU8);
  if (is_little_endian()) {
    png_set_swap(png_ptr);
  }
  auto t_ptr = (uint8_t*)tensor.data_ptr();
  for (int pass = 0; pass < number_of_passes; pass++) {
    for (png_uint_32 i = 0; i < height; ++i) {
      png_read_row(png_ptr, t_ptr, nullptr);
      t_ptr += num_pixels_per_row * (is_16_bits ? 2 : 1);
    }
    t_ptr = (uint8_t*)tensor.data_ptr();
  }

  int exif_orientation = -1;
  if (apply_exif_orientation) {
    exif_orientation = fetch_png_exif_orientation(png_ptr, info_ptr);
  }

  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);

  auto output = tensor.permute({2, 0, 1});
  if (apply_exif_orientation) {
    return exif_orientation_transform(output, exif_orientation);
  }
  return output;
}
#endif

} // namespace image
} // namespace vision
