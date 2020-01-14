// Copyright 2004-present Facebook. All Rights Reserved.

#include "seekable_buffer.h"
#include <c10/util/Logging.h>
#include <chrono>

extern "C" {
#include <libavformat/avio.h>
}

namespace ffmpeg {

bool SeekableBuffer::init(
    DecoderInCallback&& in,
    ssize_t minSize,
    ssize_t maxSize,
    uint64_t timeoutMs) {
  inCallback_ = std::forward<DecoderInCallback>(in);
  len_ = minSize;
  buffer_.resize(len_);
  pos_ = 0;
  end_ = 0;
  eof_ = 0;

  auto end =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  auto watcher = [end]() -> bool {
    return std::chrono::steady_clock::now() <= end;
  };

  bool hasTime = false;
  while (!eof_ && end_ < maxSize && (hasTime = watcher())) {
    // lets read all bytes into available buffer
    auto res = inCallback_(buffer_.data() + end_, len_ - end_, timeoutMs);
    if (res > 0) {
      end_ += res;
      if (end_ == len_) {
        len_ = std::min(len_ * 4, maxSize);
        buffer_.resize(len_);
      }
    } else if (res == 0) {
      eof_ = 1;
    } else {
      // error
      return false;
    }
  }

  if (!hasTime) {
    return false;
  }

  if (buffer_.size() > 2 && buffer_[0] == 0xFF && buffer_[1] == 0xD8 &&
      buffer_[2] == 0xFF) {
    imageType_ = ImageType::JPEG;
  } else if (
      buffer_.size() > 3 && buffer_[1] == 'P' && buffer_[2] == 'N' &&
      buffer_[3] == 'G') {
    imageType_ = ImageType::PNG;
  } else if (
      buffer_.size() > 1 &&
      ((buffer_[0] == 0x49 && buffer_[1] == 0x49) ||
       (buffer_[0] == 0x4D && buffer_[1] == 0x4D))) {
    imageType_ = ImageType::TIFF;
  }

  return true;
}

int SeekableBuffer::read(uint8_t* buf, int size, uint64_t timeoutMs) {
  // 1. pos_ < end_
  if (pos_ < end_) {
    auto available = std::min(int(end_ - pos_), size);
    memcpy(buf, buffer_.data() + pos_, available);
    pos_ += available;
    return available;
  } else if (!eof_) {
    auto res = inCallback_(buf, size, timeoutMs); // read through
    if (res > 0) {
      pos_ += res;
      if (pos_ > end_ && !buffer_.empty()) {
        std::vector<uint8_t>().swap(buffer_);
      }
    } else if (res == 0) {
      eof_ = 1;
    }
    return res;
  } else {
    return 0;
  }
}

int64_t SeekableBuffer::seek(int64_t offset, int whence, uint64_t timeoutMs) {
  // remove force flag
  whence &= ~AVSEEK_FORCE;
  // get size request
  int size = whence & AVSEEK_SIZE;
  // remove size flag
  whence &= ~AVSEEK_SIZE;

  if (size) {
    return eof_ ? end_ : AVERROR(EINVAL);
  } else {
    switch (whence) {
      case SEEK_SET:
        if (offset < 0) {
          return AVERROR(EINVAL);
        }
        if (offset <= end_) {
          pos_ = offset;
          return pos_;
        }
        if (!inCallback_(0, offset, timeoutMs)) {
          pos_ = offset;
          return 0;
        }
        break;
      case SEEK_END:
        if (eof_ && pos_ <= end_ && offset < 0 && end_ + offset >= 0) {
          pos_ = end_ + offset;
          return 0;
        }
        break;
      case SEEK_CUR:
        if (pos_ + offset < 0) {
          return AVERROR(EINVAL);
        }
        if (pos_ + offset <= end_) {
          pos_ += offset;
          return 0;
        }
        if (!inCallback_(0, pos_ + offset, timeoutMs)) {
          pos_ += offset;
          return 0;
        }
        break;
      default:
        LOG(ERROR) << "Unknown whence flag gets provided: " << whence;
    }
  }
  return AVERROR(EINVAL); // we have no idea what the media size is
}

void SeekableBuffer::shutdown() {
  inCallback_ = nullptr;
}

} // namespace ffmpeg
