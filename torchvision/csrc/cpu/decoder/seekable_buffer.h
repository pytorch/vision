// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "defs.h"

namespace ffmpeg {

/**
 * Class uses internal buffer to store initial size bytes as a seekable cache
 * from Media provider and let ffmpeg to seek and read bytes from cache
 * and beyond - reading bytes directly from Media provider
 */
enum class ImageType {
  UNKNOWN = 0,
  JPEG = 1,
  PNG = 2,
  TIFF = 3,
};

class SeekableBuffer {
 public:
  // try to fill out buffer, returns true if EOF detected (seek will supported)
  bool init(
      DecoderInCallback&& in,
      ssize_t minSize,
      ssize_t maxSize,
      uint64_t timeoutMs);
  int read(uint8_t* buf, int size, uint64_t timeoutMs);
  int64_t seek(int64_t offset, int whence, uint64_t timeoutMs);
  void shutdown();
  ImageType getImageType() const {
    return imageType_;
  }

 private:
  DecoderInCallback inCallback_;
  std::vector<uint8_t> buffer_; // resized at init time
  ssize_t len_{0}; // current buffer size
  ssize_t pos_{0}; // current position (SEEK_CUR iff pos_ < end_)
  ssize_t end_{0}; // bytes in buffer [0, buffer_.size()]
  ssize_t eof_{0}; // indicates the EOF
  ImageType imageType_{ImageType::UNKNOWN};
};

} // namespace ffmpeg
