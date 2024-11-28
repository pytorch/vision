#include "decode_image.h"

#include "decode_avif.h"
#include "decode_gif.h"
#include "decode_heic.h"
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
      "Unsupported image file. Only jpeg, png and gif are currently supported.";

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

  // We assume the signature of an avif file is
  // 0000 0020 6674 7970 6176 6966
  // xxxx xxxx  f t  y p  a v  i f
  // We only check for the "ftyp avif" part.
  // This is probably not perfect, but hopefully this should cover most files.
  const uint8_t avif_signature[8] = {
      0x66, 0x74, 0x79, 0x70, 0x61, 0x76, 0x69, 0x66}; // == "ftypavif"
  TORCH_CHECK(data.numel() >= 12, err_msg);
  if ((memcmp(avif_signature, datap + 4, 8) == 0)) {
    return decode_avif(data, mode);
  }

  // Similarly for heic we assume the signature is "ftypeheic" but some files
  // may come as "ftypmif1" where the "heic" part is defined later in the file.
  // We can't be re-inventing libmagic here. We might need to start relying on
  // it though...
  const uint8_t heic_signature[8] = {
      0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63}; // == "ftypheic"
  TORCH_CHECK(data.numel() >= 12, err_msg);
  if ((memcmp(heic_signature, datap + 4, 8) == 0)) {
    return decode_heic(data, mode);
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
