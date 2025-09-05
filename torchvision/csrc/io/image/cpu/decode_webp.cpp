#include "decode_webp.h"
#include "../common.h"

#if WEBP_FOUND
#include "webp/decode.h"
#include "webp/types.h"
#endif // WEBP_FOUND

namespace vision {
namespace image {

#if !WEBP_FOUND
torch::Tensor decode_webp(
    const torch::Tensor& encoded_data,
    ImageReadMode mode) {
  TORCH_CHECK(
      false, "decode_webp: torchvision not compiled with libwebp support");
}
#else

torch::Tensor decode_webp(
    const torch::Tensor& encoded_data,
    ImageReadMode mode) {
  validate_encoded_data(encoded_data);

  auto encoded_data_p = encoded_data.data_ptr<uint8_t>();
  auto encoded_data_size = encoded_data.numel();

  WebPBitstreamFeatures features;
  auto res = WebPGetFeatures(encoded_data_p, encoded_data_size, &features);
  TORCH_CHECK(
      res == VP8_STATUS_OK, "WebPGetFeatures failed with error code ", res);
  TORCH_CHECK(
      !features.has_animation, "Animated webp files are not supported.");

  if (mode == IMAGE_READ_MODE_GRAY || mode == IMAGE_READ_MODE_GRAY_ALPHA) {
    TORCH_WARN_ONCE(
        "Webp does not support grayscale conversions. "
        "The returned tensor will be in the colorspace of the original image.");
  }

  auto return_rgb =
      should_this_return_rgb_or_rgba_let_me_know_in_the_comments_down_below_guys_see_you_in_the_next_video(
          mode, features.has_alpha);

  auto decoding_func = return_rgb ? WebPDecodeRGB : WebPDecodeRGBA;
  auto num_channels = return_rgb ? 3 : 4;

  int width = 0;
  int height = 0;

  auto decoded_data =
      decoding_func(encoded_data_p, encoded_data_size, &width, &height);

  TORCH_CHECK(decoded_data != nullptr, "WebPDecodeRGB[A] failed.");

  auto deleter = [decoded_data](void*) { WebPFree(decoded_data); };
  auto out = torch::from_blob(
      decoded_data, {height, width, num_channels}, deleter, torch::kUInt8);

  return out.permute({2, 0, 1});
}
#endif // WEBP_FOUND

} // namespace image
} // namespace vision
