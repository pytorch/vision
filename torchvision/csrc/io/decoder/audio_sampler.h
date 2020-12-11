#pragma once

#include "defs.h"

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
  int sample(const ByteStorage* in, ByteStorage* out) override;
  void shutdown() override;

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
  SwrContext* swrContext_{nullptr};
  void* logCtx_{nullptr};
};

} // namespace ffmpeg
