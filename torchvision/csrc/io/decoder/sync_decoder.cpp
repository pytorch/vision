#include "sync_decoder.h"
#include <c10/util/Logging.h>

namespace ffmpeg {

SyncDecoder::AVByteStorage::AVByteStorage(size_t n) {
  ensure(n);
}

SyncDecoder::AVByteStorage::~AVByteStorage() {
  av_free(buffer_);
}

void SyncDecoder::AVByteStorage::ensure(size_t n) {
  if (tail() < n) {
    capacity_ = offset_ + length_ + n;
    buffer_ = static_cast<uint8_t*>(av_realloc(buffer_, capacity_));
  }
}

uint8_t* SyncDecoder::AVByteStorage::writableTail() {
  TORCH_CHECK_LE(offset_ + length_, capacity_);
  return buffer_ + offset_ + length_;
}

void SyncDecoder::AVByteStorage::append(size_t n) {
  TORCH_CHECK_LE(n, tail());
  length_ += n;
}

void SyncDecoder::AVByteStorage::trim(size_t n) {
  TORCH_CHECK_LE(n, length_);
  offset_ += n;
  length_ -= n;
}

const uint8_t* SyncDecoder::AVByteStorage::data() const {
  return buffer_ + offset_;
}

size_t SyncDecoder::AVByteStorage::length() const {
  return length_;
}

size_t SyncDecoder::AVByteStorage::tail() const {
  TORCH_CHECK_LE(offset_ + length_, capacity_);
  return capacity_ - offset_ - length_;
}

void SyncDecoder::AVByteStorage::clear() {
  offset_ = 0;
  length_ = 0;
}

std::unique_ptr<ByteStorage> SyncDecoder::createByteStorage(size_t n) {
  return std::make_unique<AVByteStorage>(n);
}

void SyncDecoder::onInit() {
  eof_ = false;
  queue_.clear();
}

int SyncDecoder::decode(DecoderOutputMessage* out, uint64_t timeoutMs) {
  if (eof_ && queue_.empty()) {
    return ENODATA;
  }

  if (queue_.empty()) {
    int result = getFrame(timeoutMs);
    // assign EOF
    eof_ = result == ENODATA;
    // check unrecoverable error, any error but ENODATA
    if (result && result != ENODATA) {
      return result;
    }

    // still empty
    if (queue_.empty()) {
      if (eof_) {
        return ENODATA;
      } else {
        LOG(INFO) << "Queue is empty";
        return ETIMEDOUT;
      }
    }
  }

  *out = std::move(queue_.front());
  queue_.pop_front();
  return 0;
}

void SyncDecoder::push(DecoderOutputMessage&& buffer) {
  queue_.push_back(std::move(buffer));
}
} // namespace ffmpeg
