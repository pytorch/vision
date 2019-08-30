// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "audio_sampler.h"
#include "stream.h"
#include "time_keeper.h"

namespace ffmpeg {

/**
 * Class uses FFMPEG library to decode one audio stream.
 */

class AudioStream : public Stream {
 public:
  AudioStream(
      AVFormatContext* inputCtx,
      int index,
      bool convertPtsToWallTime,
      const AudioFormat& format);
  ~AudioStream() override;

 private:
  // Stream overrides
  MediaType getMediaType() const override {
    return TYPE_AUDIO;
  }
  int initFormat() override;
  int estimateBytes(bool flush) override;
  int copyFrameBytes(ByteStorage* out, bool flush) override;
  void setHeader(DecoderHeader* header) override;

  void ensureSampler();

 private:
  AudioFormat format_;
  std::unique_ptr<AudioSampler> sampler_;
  TimeKeeper keeper_;
};

} // namespace ffmpeg
