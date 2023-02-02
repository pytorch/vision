#include "memory_buffer.h"
#include <c10/util/Logging.h>

namespace ffmpeg {

MemoryBuffer::MemoryBuffer(const uint8_t* buffer, size_t size)
    : buffer_(buffer), len_(size) {}

int MemoryBuffer::read(uint8_t* buf, int size) {
  if (pos_ < len_) {
    auto available = std::min(int(len_ - pos_), size);
    memcpy(buf, buffer_ + pos_, available);
    pos_ += available;
    return available;
  }

  return 0;
}

int64_t MemoryBuffer::seek(int64_t offset, int whence) {
  if (whence & AVSEEK_SIZE) {
    return len_;
  }

  // remove force flag
  whence &= ~AVSEEK_FORCE;

  switch (whence) {
    case SEEK_SET:
      if (offset >= 0 && offset <= len_) {
        pos_ = offset;
      }
      break;
    case SEEK_END:
      if (len_ + offset >= 0 && len_ + offset <= len_) {
        pos_ = len_ + offset;
      }
      break;
    case SEEK_CUR:
      if (pos_ + offset > 0 && pos_ + offset <= len_) {
        pos_ += offset;
      }
      break;
    default:
      LOG(ERROR) << "Unknown whence flag gets provided: " << whence;
  }
  return pos_;
}

/* static */
DecoderInCallback MemoryBuffer::getCallback(
    const uint8_t* buffer,
    size_t size) {
  MemoryBuffer object(buffer, size);
  return
      [object](uint8_t* out, int size, int whence, uint64_t timeoutMs) mutable
      -> int {
        if (out) { // see defs.h file
          // read mode
          return object.read(out, size);
        }
        // seek mode
        if (!timeoutMs) {
          // seek capability, yes - supported
          return 0;
        }
        return object.seek(size, whence);
      };
}

} // namespace ffmpeg
