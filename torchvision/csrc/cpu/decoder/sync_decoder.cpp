// Copyright 2004-present Facebook. All Rights Reserved.

#include "sync_decoder.h"
#include <c10/util/Logging.h>

namespace ffmpeg {

SyncDecoder::VectorByteStorage::VectorByteStorage(size_t n) {
  buffer_.resize(n);
}

void SyncDecoder::VectorByteStorage::ensure(size_t n) {
  if (tail() < n) {
    buffer_.resize(offset_ + length_ + n);
  }
}

uint8_t* SyncDecoder::VectorByteStorage::writableTail() {
  CHECK_LE(offset_ + length_, buffer_.size());
  return buffer_.data() + offset_ + length_;
}

void SyncDecoder::VectorByteStorage::append(size_t n) {
  CHECK_LE(n, tail());
  length_ += n;
}

void SyncDecoder::VectorByteStorage::trim(size_t n) {
  CHECK_LE(n, length_);
  offset_ += n;
  length_ -= n;
}

const uint8_t* SyncDecoder::VectorByteStorage::data() const {
  return buffer_.data() + offset_;
}

size_t SyncDecoder::VectorByteStorage::length() const {
  return length_;
}

size_t SyncDecoder::VectorByteStorage::tail() const {
  auto size = buffer_.size();
  CHECK_LE(offset_ + length_, buffer_.size());
  return size - offset_ - length_;
}

void SyncDecoder::VectorByteStorage::clear() {
  buffer_.clear();
  offset_ = 0;
  length_ = 0;
}

std::unique_ptr<ByteStorage> SyncDecoder::createByteStorage(size_t n) {
  return std::make_unique<VectorByteStorage>(n);
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
    int result = getBytes(timeoutMs);
    eof_ = result == ENODATA;

    if (result && result != ENODATA) {
      return result;
    }

    // still empty
    if (queue_.empty()) {
      return ETIMEDOUT;
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
