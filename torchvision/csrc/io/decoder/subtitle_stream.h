#pragma once

#include "stream.h"
#include "subtitle_sampler.h"

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
  void setFramePts(DecoderHeader* header, bool flush) override;

 private:
  int initFormat() override;
  int analyzePacket(const AVPacket* packet, bool* gotFrame) override;
  int copyFrameBytes(ByteStorage* out, bool flush) override;
  void releaseSubtitle();

 private:
  SubtitleSampler sampler_;
  AVSubtitleKeeper sub_;
};

} // namespace ffmpeg
