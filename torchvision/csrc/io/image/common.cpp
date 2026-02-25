
#include "common.h"

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
void* PyInit_image(void) {
  return nullptr;
}
#endif

namespace vision {
namespace image {

using namespace vision::stable;

void validate_encoded_data(const Tensor& encoded_data) {
  VISION_CHECK(
      encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  VISION_CHECK(
      encoded_data.scalar_type() == kByte,
      "Input tensor must have uint8 data type, got ",
      scalarTypeName(encoded_data.scalar_type()));
  VISION_CHECK(
      encoded_data.dim() == 1 && encoded_data.numel() > 0,
      "Input tensor must be 1-dimensional and non-empty, got ",
      encoded_data.dim(),
      " dims  and ",
      encoded_data.numel(),
      " numels.");
}

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

} // namespace image
} // namespace vision
