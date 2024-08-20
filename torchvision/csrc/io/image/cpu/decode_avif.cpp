#include "decode_avif.h"

#if AVIF_FOUND
#include "avif/avif.h"
#endif // AVIF_FOUND

namespace vision {
namespace image {

#if !AVIF_FOUND
torch::Tensor decode_avif(const torch::Tensor& data) {
  TORCH_CHECK(
      false, "decode_avif: torchvision not compiled with libavif support");
}
#else

// This normally comes from avif_cxx.h, but it's not always present when
// installing libavif. So we just copy/paste it here.
struct UniquePtrDeleter {
  void operator()(avifDecoder* decoder) const {
    avifDecoderDestroy(decoder);
  }
};
using DecoderPtr = std::unique_ptr<avifDecoder, UniquePtrDeleter>;

torch::Tensor decode_avif(const torch::Tensor& encoded_data) {
  // This is based on
  // https://github.com/AOMediaCodec/libavif/blob/main/examples/avif_example_decode_memory.c
  // Refer there for more detail about what each function does, and which
  // structure/data is available after which call.

  TORCH_CHECK(encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  TORCH_CHECK(
      encoded_data.dtype() == torch::kU8,
      "Input tensor must have uint8 data type, got ",
      encoded_data.dtype());
  TORCH_CHECK(
      encoded_data.dim() == 1,
      "Input tensor must be 1-dimensional, got ",
      encoded_data.dim(),
      " dims.");

  DecoderPtr decoder(avifDecoderCreate());
  TORCH_CHECK(decoder != nullptr, "Failed to create avif decoder.");

  auto result = AVIF_RESULT_UNKNOWN_ERROR;
  result = avifDecoderSetIOMemory(
      decoder.get(), encoded_data.data_ptr<uint8_t>(), encoded_data.numel());
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderSetIOMemory failed:",
      avifResultToString(result));

  result = avifDecoderParse(decoder.get());
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderParse failed: ",
      avifResultToString(result));
  TORCH_CHECK(
      decoder->imageCount == 1, "Avif file contains more than one image");
  TORCH_CHECK(
      decoder->image->depth <= 8,
      "avif images with bitdepth > 8 are not supported");

  result = avifDecoderNextImage(decoder.get());
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderNextImage failed:",
      avifResultToString(result));

  auto out = torch::empty(
      {decoder->image->height, decoder->image->width, 3}, torch::kUInt8);

  avifRGBImage rgb;
  memset(&rgb, 0, sizeof(rgb));
  avifRGBImageSetDefaults(&rgb, decoder->image);
  rgb.format = AVIF_RGB_FORMAT_RGB;
  rgb.pixels = out.data_ptr<uint8_t>();
  rgb.rowBytes = rgb.width * avifRGBImagePixelSize(&rgb);

  result = avifImageYUVToRGB(decoder->image, &rgb);
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifImageYUVToRGB failed: ",
      avifResultToString(result));

  return out.permute({2, 0, 1}); // return CHW, channels-last
}
#endif // AVIF_FOUND

} // namespace image
} // namespace vision
