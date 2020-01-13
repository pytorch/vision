// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "decoder.h"
#include <list>

namespace ffmpeg {

/**
 * Class uses FFMPEG library to decode media streams.
 * Media bytes can be explicitly provided through read-callback
 * or fetched internally by FFMPEG library
 */
class SyncDecoder : public Decoder {

  class VectorByteStorage : public ByteStorage {
   public:
     VectorByteStorage(size_t n);
     void ensure(size_t n) override;
     uint8_t* writableTail() override;
     void append(size_t n) override;
     void trim(size_t n) override;
     const uint8_t* data() const override;
     size_t length() const override;
     size_t tail() const override;
     void clear() override;
    private:
     size_t offset_{0};
     size_t length_{0};
     std::vector<uint8_t> buffer_;
  };

 public:
  int decode(DecoderOutputMessage* out, uint64_t timeoutMs) override;

 private:
  void push(DecoderOutputMessage&& buffer) override;
  void onInit() override;
  std::unique_ptr<ByteStorage> createByteStorage(size_t n) override;

 private:
  std::list<DecoderOutputMessage> queue_;
  bool eof_{false};
};
}
