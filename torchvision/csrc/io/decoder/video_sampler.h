#pragma once

#include "defs.h"

namespace ffmpeg {

/**
 * Class transcode video frames from one format into another
 */

class VideoSampler : public MediaSampler {
 public:
  VideoSampler(int swsFlags = SWS_AREA, int64_t loggingUuid = 0);

  ~VideoSampler() override;

  // MediaSampler overrides
  bool init(const SamplerParameters& params) override;
  int sample(const ByteStorage* in, ByteStorage* out) override;
  void shutdown() override;

  // returns number processed/scaling bytes
  int sample(AVFrame* frame, ByteStorage* out);
  int getImageBytes() const;

 private:
  // close resources
  void cleanUp();
  // helper functions for rescaling, cropping, etc.
  int sample(
      const uint8_t* const srcSlice[],
      int srcStride[],
      ByteStorage* out);

 private:
  VideoFormat scaleFormat_;
  SwsContext* scaleContext_{nullptr};
  SwsContext* cropContext_{nullptr};
  int swsFlags_{SWS_AREA};
  std::vector<uint8_t> scaleBuffer_;
  int64_t loggingUuid_{0};
};

} // namespace ffmpeg
