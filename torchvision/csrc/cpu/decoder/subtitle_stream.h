// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "stream.h"
#include "subtitle_sampler.h"
#include "time_keeper.h"

namespace ffmpeg {

/**
 * Class uses FFMPEG library to decode one subtitle stream.
 */
struct AVSubtitleKeeper : AVSubtitle {
  int64_t release{0};
};

class SubtitleStream : public Stream {
 public:
  SubtitleStream(
      AVFormatContext* inputCtx,
      int index,
      bool convertPtsToWallTime,
      const SubtitleFormat& format);
  ~SubtitleStream() override;

 protected:
  void setHeader(DecoderHeader* header) override;

 private:
  int initFormat() override;
  int analyzePacket(const AVPacket* packet, int* gotFramePtr) override;
  int estimateBytes(bool flush) override;
  int copyFrameBytes(ByteStorage* out, bool flush) override;
  void releaseSubtitle();

 private:
  SubtitleSampler sampler_;
  TimeKeeper keeper_;
  AVSubtitleKeeper sub_;
};

} // namespace ffmpeg
