#include "decode_webp.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

#include "../common_stable.h"

#if WEBP_FOUND
#include "webp/decode.h"
#include "webp/types.h"
#endif // WEBP_FOUND

namespace vision {
namespace image {

#if !WEBP_FOUND
torch::stable::Tensor decode_webp(
    const torch::stable::Tensor& encoded_data,
    ImageReadMode mode) {
  STD_TORCH_CHECK(
      false, "decode_webp: torchvision not compiled with libwebp support");
}
#else

namespace {
bool should_this_return_rgb_or_rgba_let_me_know_in_the_comments_down_below_guys_see_you_in_the_next_video(
    ImageReadMode mode,
    bool has_alpha) {
  // Return true if the calling decoding function should return a 3D RGB tensor,
  // and false if it should return a 4D RGBA tensor.
  // This function ignores the requested "grayscale" modes and treats it as
  // "unchanged", so it should only used on decoders who don't support grayscale
  // outputs.

  if (mode == IMAGE_READ_MODE_RGB) {
    return true;
  }
  if (mode == IMAGE_READ_MODE_RGB_ALPHA) {
    return false;
  }
  // From here we assume mode is "unchanged", even for grayscale ones.
  return !has_alpha;
}
} // namespace

torch::stable::Tensor decode_webp(
    const torch::stable::Tensor& encoded_data,
    ImageReadMode mode) {
  validate_encoded_data(encoded_data);

  auto encoded_data_p = encoded_data.const_data_ptr<uint8_t>();
  auto encoded_data_size = encoded_data.numel();

  WebPBitstreamFeatures features;
  auto res = WebPGetFeatures(encoded_data_p, encoded_data_size, &features);
  STD_TORCH_CHECK(
      res == VP8_STATUS_OK, "WebPGetFeatures failed with error code ", res);
  STD_TORCH_CHECK(
      !features.has_animation, "Animated webp files are not supported.");

  if (mode == IMAGE_READ_MODE_GRAY || mode == IMAGE_READ_MODE_GRAY_ALPHA) {
    // TODO(stable-abi): warn once per process currently not available, replace
    // when a stable warn-once lands.
    aoti_torch_warn(
        __func__,
        __FILE__,
        static_cast<uint32_t>(__LINE__),
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

  STD_TORCH_CHECK(decoded_data != nullptr, "WebPDecodeRGB[A] failed.");

  auto deleter = [decoded_data](void*) { WebPFree(decoded_data); };
  auto out = torch::stable::from_blob(
      decoded_data,
      {height, width, num_channels},
      {width * num_channels, num_channels, 1},
      torch::stable::Device(torch::headeronly::DeviceType::CPU),
      torch::headeronly::ScalarType::Byte,
      deleter);

  return stable_permute(out, {2, 0, 1});
}
#endif // WEBP_FOUND

STABLE_TORCH_LIBRARY_FRAGMENT(image, m) {
  m.def("decode_webp(Tensor encoded_data, int mode) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(image, CompositeExplicitAutograd, m) {
  m.impl("decode_webp", TORCH_BOX(&decode_webp));
}

} // namespace image
} // namespace vision
