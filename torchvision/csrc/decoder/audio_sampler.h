// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "defs.h"

extern "C" {
#include <libswresample/swresample.h>
}

namespace ffmpeg {

/**
 * Class transcode audio frames from one format into another
 */

class AudioSampler : public MediaSampler {
 public:
  explicit AudioSampler(void* logCtx);
  ~AudioSampler() override;

  // MediaSampler overrides
  bool init(const SamplerParameters& params) override;
  MediaType getMediaType() const override {
    return MediaType::TYPE_AUDIO;
  }
  FormatUnion getInputFormat() const override {
    return params_.in;
  }
  FormatUnion getOutFormat() const override {
    return params_.out;
  }
  int sample(const ByteStorage* in, ByteStorage* out) override;
  void shutdown() override;

  int getSamplesBytes(AVFrame* frame) const;
  int sample(AVFrame* frame, ByteStorage* out);

 private:
  // close resources
  void cleanUp();
  // helper functions for rescaling, cropping, etc.
  int numOutputSamples(int inSamples) const;
  int sample(
      const uint8_t* inPlanes[],
      int inNumSamples,
      ByteStorage* out,
      int outNumSamples);

 private:
  SamplerParameters params_;
  SwrContext* swrContext_{nullptr};
  void* logCtx_{nullptr};
};

} // namespace ffmpeg
