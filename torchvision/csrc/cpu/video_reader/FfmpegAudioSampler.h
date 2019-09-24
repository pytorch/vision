#pragma once

#include "FfmpegSampler.h"

#define AVRESAMPLE_MAX_CHANNELS 32

/**
 * Class transcode audio frames from one format into another
 */
class FfmpegAudioSampler : public FfmpegSampler {
 public:
  explicit FfmpegAudioSampler(const AudioFormat& in, const AudioFormat& out);
  ~FfmpegAudioSampler() override;

  int init() override;

  int64_t getSampleBytes(const AVFrame* frame) const;
  // FfmpegSampler overrides
  // returns number of bytes of the sampled data
  std::unique_ptr<DecodedFrame> sample(const AVFrame* frame) override;

  const AudioFormat& getInFormat() const {
    return inFormat_;
  }

 private:
  int64_t getOutNumSamples(int inNumSamples) const;

  AudioFormat inFormat_;
  AudioFormat outFormat_;
  SwrContext* swrContext_{nullptr};
};
