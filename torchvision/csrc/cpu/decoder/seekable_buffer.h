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
  // @type is optional, not nullptr only is image detection required
  // \returns 1 is buffer seekable, 0 - if not seekable, < 0 on error
  int init(
      DecoderInCallback&& in,
      uint64_t timeoutMs,
      size_t maxSeekableBytes,
      ImageType* type);
  int read(uint8_t* buf, int size, uint64_t timeoutMs);
  int64_t seek(int64_t offset, int whence, uint64_t timeoutMs);
  void shutdown();

 private:
  bool readBytes(DecoderInCallback& in, size_t maxBytes, uint64_t timeoutMs);
  void setImageType(ImageType* type);

 private:
  DecoderInCallback inCallback_;
  std::vector<uint8_t> buffer_; // resized at init time
  long pos_{0}; // current position (SEEK_CUR iff pos_ < end_)
  long end_{0}; // current buffer size
  bool eof_{0}; // indicates the EOF
  bool isSeekable_{false}; // is callback seekable
};

} // namespace ffmpeg
