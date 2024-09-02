#include "decode_avif.h"
#include "../common.h"

#if AVIF_FOUND
#include "avif/avif.h"
#endif // AVIF_FOUND

namespace vision {
namespace image {

#if !AVIF_FOUND
torch::Tensor decode_avif(
    const torch::Tensor& encoded_data,
    ImageReadMode mode) {
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

torch::Tensor decode_avif(
    const torch::Tensor& encoded_data,
    ImageReadMode mode) {
  // This is based on
  // https://github.com/AOMediaCodec/libavif/blob/main/examples/avif_example_decode_memory.c
  // Refer there for more detail about what each function does, and which
  // structure/data is available after which call.

  validate_encoded_data(encoded_data);

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

  result = avifDecoderNextImage(decoder.get());
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderNextImage failed:",
      avifResultToString(result));

  avifRGBImage rgb;
  memset(&rgb, 0, sizeof(rgb));
  avifRGBImageSetDefaults(&rgb, decoder->image);

  // images encoded as 10 or 12 bits will be decoded as uint16. The rest are
  // decoded as uint8.
  auto use_uint8 = (decoder->image->depth <= 8);
  rgb.depth = use_uint8 ? 8 : 16;

  auto return_rgb =
      should_this_return_rgb_or_rgba_let_me_know_in_the_comments_down_below_guys_see_you_in_the_next_video(
          mode, decoder->alphaPresent);

  auto num_channels = return_rgb ? 3 : 4;
  rgb.format = return_rgb ? AVIF_RGB_FORMAT_RGB : AVIF_RGB_FORMAT_RGBA;
  rgb.ignoreAlpha = return_rgb ? AVIF_TRUE : AVIF_FALSE;

  auto out = torch::empty(
      {rgb.height, rgb.width, num_channels},
      use_uint8 ? torch::kUInt8 : at::kUInt16);
  rgb.pixels = (uint8_t*)out.data_ptr();
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
