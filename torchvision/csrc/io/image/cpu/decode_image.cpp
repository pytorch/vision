#include "decode_image.h"

#include "decode_gif.h"
#include "decode_jpeg.h"
#include "decode_png.h"
#include "decode_webp.h"

namespace vision {
namespace image {

torch::Tensor decode_image(
    const torch::Tensor& data,
    ImageReadMode mode,
    bool apply_exif_orientation) {
  // Check that tensor is a CPU tensor
  TORCH_CHECK(data.device() == torch::kCPU, "Expected a CPU tensor");
  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  auto err_msg =
      "Unsupported image file. Only jpeg, png, webp and gif are currently supported. For avif and heic format, please rely on `decode_avif` and `decode_heic` directly.";

  auto datap = data.data_ptr<uint8_t>();

  const uint8_t jpeg_signature[3] = {255, 216, 255}; // == "\xFF\xD8\xFF"
  TORCH_CHECK(data.numel() >= 3, err_msg);
  if (memcmp(jpeg_signature, datap, 3) == 0) {
    return decode_jpeg(data, mode, apply_exif_orientation);
  }

  const uint8_t png_signature[4] = {137, 80, 78, 71}; // == "\211PNG"
  TORCH_CHECK(data.numel() >= 4, err_msg);
  if (memcmp(png_signature, datap, 4) == 0) {
    return decode_png(data, mode, apply_exif_orientation);
  }

  const uint8_t gif_signature_1[6] = {
      0x47, 0x49, 0x46, 0x38, 0x39, 0x61}; // == "GIF89a"
  const uint8_t gif_signature_2[6] = {
      0x47, 0x49, 0x46, 0x38, 0x37, 0x61}; // == "GIF87a"
  TORCH_CHECK(data.numel() >= 6, err_msg);
  if (memcmp(gif_signature_1, datap, 6) == 0 ||
      memcmp(gif_signature_2, datap, 6) == 0) {
    return decode_gif(data);
  }

  const uint8_t webp_signature_begin[4] = {0x52, 0x49, 0x46, 0x46}; // == "RIFF"
  const uint8_t webp_signature_end[7] = {
      0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38}; // == "WEBPVP8"
  TORCH_CHECK(data.numel() >= 15, err_msg);
  if ((memcmp(webp_signature_begin, datap, 4) == 0) &&
      (memcmp(webp_signature_end, datap + 8, 7) == 0)) {
    return decode_webp(data, mode);
  }

  TORCH_CHECK(false, err_msg);
}

} // namespace image
} // namespace vision
