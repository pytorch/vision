#include "seekable_buffer.h"
#include <c10/util/Logging.h>
#include <chrono>
#include "memory_buffer.h"

namespace ffmpeg {

int SeekableBuffer::init(
    DecoderInCallback&& in,
    uint64_t timeoutMs,
    size_t maxSeekableBytes,
    ImageType* type) {
  shutdown();
  isSeekable_ = in(nullptr, 0, 0, 0) == 0;
  if (isSeekable_) { // seekable
    if (type) {
      if (!readBytes(in, 8, timeoutMs)) {
        return -1;
      }
      setImageType(type);
      end_ = 0;
      eof_ = false;
      std::vector<uint8_t>().swap(buffer_);
      // reset callback
      if (in(nullptr, 0, SEEK_SET, timeoutMs)) {
        return -1;
      }
    }
    inCallback_ = std::forward<DecoderInCallback>(in);
    return 1;
  }

  if (!readBytes(in, maxSeekableBytes + (type ? 8 : 0), timeoutMs)) {
    return -1;
  }

  if (type) {
    setImageType(type);
  }

  if (eof_) {
    end_ = 0;
    eof_ = false;
    // reuse MemoryBuffer functionality
    inCallback_ = MemoryBuffer::getCallback(buffer_.data(), buffer_.size());
    isSeekable_ = true;
    return 1;
  }
  inCallback_ = std::forward<DecoderInCallback>(in);
  return 0;
}

bool SeekableBuffer::readBytes(
    DecoderInCallback& in,
    size_t maxBytes,
    uint64_t timeoutMs) {
  // Resize to th minimum 4K page or less
  buffer_.resize(std::min(maxBytes, size_t(4 * 1024UL)));
  end_ = 0;
  eof_ = false;

  auto end =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  auto watcher = [end]() -> bool {
    return std::chrono::steady_clock::now() <= end;
  };

  bool hasTime = true;
  while (!eof_ && end_ < maxBytes && (hasTime = watcher())) {
    // lets read all bytes into available buffer
    auto res = in(buffer_.data() + end_, buffer_.size() - end_, 0, timeoutMs);
    if (res > 0) {
      end_ += res;
      if (end_ == buffer_.size()) {
        buffer_.resize(std::min(size_t(end_ * 4UL), maxBytes));
      }
    } else if (res == 0) {
      eof_ = true;
    } else {
      // error
      return false;
    }
  }

  buffer_.resize(end_);

  return hasTime;
}

void SeekableBuffer::setImageType(ImageType* type) {
  if (buffer_.size() > 2 && buffer_[0] == 0xFF && buffer_[1] == 0xD8 &&
      buffer_[2] == 0xFF) {
    *type = ImageType::JPEG;
  } else if (
      buffer_.size() > 3 && buffer_[1] == 'P' && buffer_[2] == 'N' &&
      buffer_[3] == 'G') {
    *type = ImageType::PNG;
  } else if (
      buffer_.size() > 1 &&
      ((buffer_[0] == 0x49 && buffer_[1] == 0x49) ||
       (buffer_[0] == 0x4D && buffer_[1] == 0x4D))) {
    *type = ImageType::TIFF;
  } else {
    *type = ImageType::UNKNOWN;
  }
}

int SeekableBuffer::read(uint8_t* buf, int size, uint64_t timeoutMs) {
  if (isSeekable_) {
    return inCallback_(buf, size, 0, timeoutMs);
  }
  if (pos_ < end_) {
    // read cached bytes for non-seekable callback
    auto available = std::min(int(end_ - pos_), size);
    memcpy(buf, buffer_.data() + pos_, available);
    pos_ += available;
    return available;
  } else if (!eof_) {
    // normal sequential read (see defs.h file), i.e. @buf != null
    auto res = inCallback_(buf, size, 0, timeoutMs); // read through
    eof_ = res == 0;
    return res;
  } else {
    return 0;
  }
}

int64_t SeekableBuffer::seek(int64_t offset, int whence, uint64_t timeoutMs) {
  return inCallback_(nullptr, offset, whence, timeoutMs);
}

void SeekableBuffer::shutdown() {
  pos_ = end_ = 0;
  eof_ = false;
  std::vector<uint8_t>().swap(buffer_);
  inCallback_ = nullptr;
}

} // namespace ffmpeg
