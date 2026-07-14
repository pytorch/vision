#include "decode_image.h"

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/util/Exception.h>

#include <cstring>

#include "decode_png.h"

namespace vision {
namespace image {

namespace {

// Shims over the legacy image::decode_jpeg, decode_gif and decode_webp ops
// not yet on the stable ABI.
// TODO(stable-abi): remove each shim once its decoder is ported.
torch::stable::Tensor decode_jpeg(
    const torch::stable::Tensor& data,
    ImageReadMode mode,
    bool apply_exif_orientation) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(data),
      torch::stable::detail::from(mode),
      torch::stable::detail::from(apply_exif_orientation)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "image::decode_jpeg", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

torch::stable::Tensor decode_gif(const torch::stable::Tensor& data) {
  const auto num_args = 1;
  std::array<StableIValue, num_args> stack{torch::stable::detail::from(data)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "image::decode_gif", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

torch::stable::Tensor decode_webp(
    const torch::stable::Tensor& data,
    ImageReadMode mode) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(data), torch::stable::detail::from(mode)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "image::decode_webp", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

} // namespace

torch::stable::Tensor decode_image(
    const torch::stable::Tensor& data,
    ImageReadMode mode,
    bool apply_exif_orientation) {
  // Check that tensor is a CPU tensor
  STD_TORCH_CHECK(data.is_cpu(), "Expected a CPU tensor");
  // Check that the input tensor dtype is uint8
  STD_TORCH_CHECK(
      data.scalar_type() == torch::headeronly::ScalarType::Byte,
      "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  STD_TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  auto err_msg =
      "Unsupported image file. Only jpeg, png, webp and gif are currently supported. For avif and heic format, please rely on `decode_avif` and `decode_heic` directly.";

  auto datap = data.const_data_ptr<uint8_t>();

  const uint8_t jpeg_signature[3] = {255, 216, 255}; // == "\xFF\xD8\xFF"
  STD_TORCH_CHECK(data.numel() >= 3, err_msg);
  if (memcmp(jpeg_signature, datap, 3) == 0) {
    return decode_jpeg(data, mode, apply_exif_orientation);
  }

  const uint8_t png_signature[4] = {137, 80, 78, 71}; // == "\211PNG"
  STD_TORCH_CHECK(data.numel() >= 4, err_msg);
  if (memcmp(png_signature, datap, 4) == 0) {
    return decode_png(data, mode, apply_exif_orientation);
  }

  const uint8_t gif_signature_1[6] = {
      0x47, 0x49, 0x46, 0x38, 0x39, 0x61}; // == "GIF89a"
  const uint8_t gif_signature_2[6] = {
      0x47, 0x49, 0x46, 0x38, 0x37, 0x61}; // == "GIF87a"
  STD_TORCH_CHECK(data.numel() >= 6, err_msg);
  if (memcmp(gif_signature_1, datap, 6) == 0 ||
      memcmp(gif_signature_2, datap, 6) == 0) {
    return decode_gif(data);
  }

  const uint8_t webp_signature_begin[4] = {0x52, 0x49, 0x46, 0x46}; // == "RIFF"
  const uint8_t webp_signature_end[7] = {
      0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38}; // == "WEBPVP8"
  STD_TORCH_CHECK(data.numel() >= 15, err_msg);
  if ((memcmp(webp_signature_begin, datap, 4) == 0) &&
      (memcmp(webp_signature_end, datap + 8, 7) == 0)) {
    return decode_webp(data, mode);
  }

  STD_TORCH_CHECK(false, err_msg);
}

STABLE_TORCH_LIBRARY_FRAGMENT(image, m) {
  m.def(
      "decode_image(Tensor data, int mode, bool apply_exif_orientation=False) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(image, CompositeExplicitAutograd, m) {
  m.impl("decode_image", TORCH_BOX(&decode_image));
}

} // namespace image
} // namespace vision
