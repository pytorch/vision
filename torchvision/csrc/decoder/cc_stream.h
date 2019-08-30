// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "subtitle_stream.h"

namespace ffmpeg {

/**
 * Class uses FFMPEG library to decode one closed captions stream.
 */
class CCStream : public SubtitleStream {
 public:
  using SubtitleStream::SubtitleStream;

 private:
  // Stream overrides
  MediaType getMediaType() const override {
    return TYPE_CC;
  }

  void setHeader(DecoderHeader* header) override;
  AVCodec* findCodec(AVCodecContext* ctx) override;
};

} // namespace ffmpeg
