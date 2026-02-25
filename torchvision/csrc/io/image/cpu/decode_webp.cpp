#include "decode_webp.h"
#include "../common.h"

#if WEBP_FOUND
#include "webp/decode.h"
#include "webp/types.h"
#endif // WEBP_FOUND

namespace vision {
namespace image {

using namespace vision::stable;

#if !WEBP_FOUND
Tensor decode_webp(const Tensor& encoded_data, ImageReadMode mode) {
  VISION_CHECK(
      false, "decode_webp: torchvision not compiled with libwebp support");
}
#else

Tensor decode_webp(const Tensor& encoded_data, ImageReadMode mode) {
  validate_encoded_data(encoded_data);

  auto encoded_data_p = encoded_data.const_data_ptr<uint8_t>();
  auto encoded_data_size = encoded_data.numel();

  WebPBitstreamFeatures features;
  auto res = WebPGetFeatures(encoded_data_p, encoded_data_size, &features);
  VISION_CHECK(
      res == VP8_STATUS_OK, "WebPGetFeatures failed with error code ", res);
  VISION_CHECK(
      !features.has_animation, "Animated webp files are not supported.");

  // TODO_STABLE_ABI: need TORCH_WARN_ONCE
  //   if (mode == IMAGE_READ_MODE_GRAY || mode == IMAGE_READ_MODE_GRAY_ALPHA) {
  //     TORCH_WARN_ONCE(
  //         "Webp does not support grayscale conversions. "
  //         "The returned tensor will be in the colorspace of the original
  //         image.");
  //   }

  auto return_rgb =
      should_this_return_rgb_or_rgba_let_me_know_in_the_comments_down_below_guys_see_you_in_the_next_video(
          mode, features.has_alpha);

  auto decoding_func = return_rgb ? WebPDecodeRGB : WebPDecodeRGBA;
  auto num_channels = return_rgb ? 3 : 4;

  int width = 0;
  int height = 0;

  auto decoded_data =
      decoding_func(encoded_data_p, encoded_data_size, &width, &height);

  VISION_CHECK(decoded_data != nullptr, "WebPDecodeRGB[A] failed.");

  // Create tensor and copy data (from_blob with deleter not available in stable
  // ABI)
  auto out = emptyCPU({height, width, num_channels}, kByte);
  auto out_ptr = out.mutable_data_ptr<uint8_t>();
  std::memcpy(out_ptr, decoded_data, height * width * num_channels);

  // Free the webp-allocated memory
  WebPFree(decoded_data);

  return permute(out, {2, 0, 1});
}
#endif // WEBP_FOUND

} // namespace image
} // namespace vision
