#include "decode_image.h"

#include "decode_gif.h"
#include "decode_jpeg.h"
#include "decode_png.h"

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

  auto datap = data.data_ptr<uint8_t>();

  const uint8_t jpeg_signature[3] = {255, 216, 255}; // == "\xFF\xD8\xFF"
  const uint8_t png_signature[4] = {137, 80, 78, 71}; // == "\211PNG"
  const uint8_t gif_signature_1[6] = {
      0x47, 0x49, 0x46, 0x38, 0x39, 0x61}; // == "GIF89a"
  const uint8_t gif_signature_2[6] = {
      0x47, 0x49, 0x46, 0x38, 0x37, 0x61}; // == "GIF87a"

  if (memcmp(jpeg_signature, datap, 3) == 0) {
    return decode_jpeg(data, mode, apply_exif_orientation);
  } else if (memcmp(png_signature, datap, 4) == 0) {
    return decode_png(
        data, mode, /*allow_16_bits=*/false, apply_exif_orientation);
  } else if (
      memcmp(gif_signature_1, datap, 6) == 0 ||
      memcmp(gif_signature_2, datap, 6) == 0) {
    return decode_gif(data);
  } else {
    TORCH_CHECK(
        false,
        "Unsupported image file. Only jpeg, png and gif ",
        "are currently supported.");
  }
}

} // namespace image
} // namespace vision
