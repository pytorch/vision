/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#pragma once
// Functions in this module are taken from OpenCV
// https://github.com/opencv/opencv/blob/097891e311fae1d8354eb092a0fd0171e630d78c/modules/imgcodecs/src/exif.cpp

#if JPEG_FOUND
#include <jpeglib.h>
#endif
#if PNG_FOUND
#include <png.h>
#endif

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

class ExifDataReader {
 public:
  ExifDataReader(unsigned char* p, size_t s) : _ptr(p), _size(s) {}
  size_t size() const {
    return _size;
  }
  const unsigned char& operator[](size_t index) const {
    TORCH_CHECK(index >= 0 && index < _size);
    return _ptr[index];
  }

 protected:
  unsigned char* _ptr;
  size_t _size;
};

inline uint16_t get_endianness(const ExifDataReader& exif_data) {
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
    const ExifDataReader& exif_data,
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
    const ExifDataReader& exif_data,
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

inline int fetch_exif_orientation(unsigned char* exif_data_ptr, size_t size) {
  int exif_orientation = -1;

  // Exif binary structure looks like this
  // First 6 bytes: [E, x, i, f, 0, 0]
  // Endianness, 2 bytes : [M, M] or [I, I]
  // Tag mark, 2 bytes: [0, 0x2a]
  // Offset, 4 bytes
  // Num entries, 2 bytes
  // Tag entries and data, tag has 2 bytes and its data has 10 bytes
  // For more details:
  // http://www.media.mit.edu/pia/Research/deepview/exif.html

  ExifDataReader exif_data(exif_data_ptr, size);
  auto endianness = get_endianness(exif_data);

  // Checking whether Tag Mark (0x002A) correspond to one contained in the
  // Jpeg file
  uint16_t tag_mark = get_uint16(exif_data, endianness, 2);
  if (tag_mark == REQ_EXIF_TAG_MARK) {
    auto offset = get_uint32(exif_data, endianness, 4);
    size_t num_entry = get_uint16(exif_data, endianness, offset);
    offset += 2; // go to start of tag fields
    constexpr size_t tiff_field_size = 12;
    for (size_t entry = 0; entry < num_entry; entry++) {
      // Here we just search for orientation tag and parse it
      auto tag_num = get_uint16(exif_data, endianness, offset);
      if (tag_num == INCORRECT_TAG) {
        break;
      }
      if (tag_num == ORIENTATION_EXIF_TAG) {
        exif_orientation = get_uint16(exif_data, endianness, offset + 8);
        break;
      }
      offset += tiff_field_size;
    }
  }
  return exif_orientation;
}

#if JPEG_FOUND
inline int fetch_jpeg_exif_orientation(j_decompress_ptr cinfo) {
  // Check for Exif marker APP1
  jpeg_saved_marker_ptr exif_marker = 0;
  jpeg_saved_marker_ptr cmarker = cinfo->marker_list;
  while (cmarker && exif_marker == 0) {
    if (cmarker->marker == APP1) {
      exif_marker = cmarker;
    }
    cmarker = cmarker->next;
  }

  if (!exif_marker) {
    return -1;
  }

  constexpr size_t start_offset = 6;
  if (exif_marker->data_length <= start_offset) {
    return -1;
  }

  auto* exif_data_ptr = exif_marker->data + start_offset;
  auto size = exif_marker->data_length - start_offset;

  return fetch_exif_orientation(exif_data_ptr, size);
}
#endif // #if JPEG_FOUND

#if PNG_FOUND && defined(PNG_eXIf_SUPPORTED)
inline int fetch_png_exif_orientation(png_structp png_ptr, png_infop info_ptr) {
  png_uint_32 num_exif = 0;
  png_bytep exif = 0;

  // Exif info could be in info_ptr
  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_eXIf)) {
    png_get_eXIf_1(png_ptr, info_ptr, &num_exif, &exif);
  }

  if (exif && num_exif > 0) {
    return fetch_exif_orientation(exif, num_exif);
  }
  return -1;
}
#endif // #if PNG_FOUND && defined(PNG_eXIf_SUPPORTED)

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
