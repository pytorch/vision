#pragma once

#include "audio_sampler.h"
#include "stream.h"

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
  int initFormat() override;
  int copyFrameBytes(ByteStorage* out, bool flush) override;

 private:
  std::unique_ptr<AudioSampler> sampler_;
};

} // namespace ffmpeg
