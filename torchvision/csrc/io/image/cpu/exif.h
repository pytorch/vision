#pragma once
#include <torch/types.h>

namespace vision {
namespace image {
namespace exif_private {

constexpr uint16_t APP1 = 0xe1;
constexpr uint16_t ENDIANNESS_INTEL = 0x49;
constexpr uint16_t ENDIANNESS_MOTO = 0x4d;
constexpr uint16_t REQ_EXIF_TAG_MARK = 0x2a;
constexpr uint16_t ORIENTATION_EXIF_TAG = 0x0112;
constexpr uint16_t INCORRECT_TAG = -1;

// Functions in this module are taken from OpenCV
// https://github.com/opencv/opencv/blob/097891e311fae1d8354eb092a0fd0171e630d78c/modules/modules/imgcodecs/src/exif.cpp
inline uint16_t get_endianness(const std::vector<unsigned char>& exif_data) {
  if ((exif_data.size() < 1) ||
      (exif_data.size() > 1 && exif_data[0] != exif_data[1])) {
    return 0;
  }
  if (exif_data[0] == 'I') {
    return ENDIANNESS_INTEL;
  }
  if (exif_data[0] == 'M') {
    return ENDIANNESS_MOTO;
  }
  return 0;
}

inline uint16_t get_uint16(
    const std::vector<unsigned char>& exif_data,
    uint16_t endianness,
    const size_t offset) {
  if (offset + 1 >= exif_data.size()) {
    return INCORRECT_TAG;
  }

  if (endianness == ENDIANNESS_INTEL) {
    return exif_data[offset] + (exif_data[offset + 1] << 8);
  }
  return (exif_data[offset] << 8) + exif_data[offset + 1];
}

inline uint32_t get_uint32(
    const std::vector<unsigned char>& exif_data,
    uint16_t endianness,
    const size_t offset) {
  if (offset + 3 >= exif_data.size()) {
    return INCORRECT_TAG;
  }

  if (endianness == ENDIANNESS_INTEL) {
    return exif_data[offset] + (exif_data[offset + 1] << 8) +
        (exif_data[offset + 2] << 16) + (exif_data[offset + 3] << 24);
  }
  return (exif_data[offset] << 24) + (exif_data[offset + 1] << 16) +
      (exif_data[offset + 2] << 8) + exif_data[offset + 3];
}

constexpr uint16_t IMAGE_ORIENTATION_TL = 1; // normal orientation
constexpr uint16_t IMAGE_ORIENTATION_TR = 2; // needs horizontal flip
constexpr uint16_t IMAGE_ORIENTATION_BR = 3; // needs 180 rotation
constexpr uint16_t IMAGE_ORIENTATION_BL = 4; // needs vertical flip
constexpr uint16_t IMAGE_ORIENTATION_LT =
    5; // mirrored horizontal & rotate 270 CW
constexpr uint16_t IMAGE_ORIENTATION_RT = 6; // rotate 90 CW
constexpr uint16_t IMAGE_ORIENTATION_RB =
    7; // mirrored horizontal & rotate 90 CW
constexpr uint16_t IMAGE_ORIENTATION_LB = 8; // needs 270 CW rotation

inline torch::Tensor exif_orientation_transform(
    const torch::Tensor& image,
    int orientation) {
  if (orientation == IMAGE_ORIENTATION_TL) {
    return image;
  } else if (orientation == IMAGE_ORIENTATION_TR) {
    return image.flip(-1);
  } else if (orientation == IMAGE_ORIENTATION_BR) {
    // needs 180 rotation equivalent to
    // flip both horizontally and vertically
    return image.flip({-2, -1});
  } else if (orientation == IMAGE_ORIENTATION_BL) {
    return image.flip(-2);
  } else if (orientation == IMAGE_ORIENTATION_LT) {
    return image.transpose(-1, -2);
  } else if (orientation == IMAGE_ORIENTATION_RT) {
    return image.transpose(-1, -2).flip(-1);
  } else if (orientation == IMAGE_ORIENTATION_RB) {
    return image.transpose(-1, -2).flip({-2, -1});
  } else if (orientation == IMAGE_ORIENTATION_LB) {
    return image.transpose(-1, -2).flip(-2);
  }
  return image;
}

} // namespace exif_private
} // namespace image
} // namespace vision
