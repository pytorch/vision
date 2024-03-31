#pragma once

#include "defs.h"

namespace ffmpeg {

/**
 * Class uses external memory buffer and implements a seekable interface.
 */
class MemoryBuffer {
 public:
  explicit MemoryBuffer(const uint8_t* buffer, size_t size);
  int64_t seek(int64_t offset, int whence);
  int read(uint8_t* buf, int size);

  // static constructor for decoder callback.
  static DecoderInCallback getCallback(const uint8_t* buffer, size_t size);

 private:
  const uint8_t* buffer_; // set at construction time
  long pos_{0}; // current position
  long len_{0}; // bytes in buffer
};

} // namespace ffmpeg
