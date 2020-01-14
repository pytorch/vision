// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "subtitle_stream.h"

namespace ffmpeg {

/**
 * Class uses FFMPEG library to decode one closed captions stream.
 */
class CCStream : public SubtitleStream {
 public:
  CCStream(
      AVFormatContext* inputCtx,
      int index,
      bool convertPtsToWallTime,
      const SubtitleFormat& format);

 private:
  AVCodec* findCodec(AVCodecContext* ctx) override;
};

} // namespace ffmpeg
