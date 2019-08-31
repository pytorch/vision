#pragma once

#include "FfmpegSampler.h"

/**
 * Class transcode video frames from one format into another
 */

class FfmpegVideoSampler : public FfmpegSampler {
 public:
  explicit FfmpegVideoSampler(
      const VideoFormat& in,
      const VideoFormat& out,
      int swsFlags = SWS_AREA);
  ~FfmpegVideoSampler() override;

  int init() override;

  int32_t getImageBytes() const;
  // returns number of bytes of the sampled data
  std::unique_ptr<DecodedFrame> sample(const AVFrame* frame) override;

  const VideoFormat& getInFormat() const {
    return inFormat_;
  }

 private:
  VideoFormat inFormat_;
  VideoFormat outFormat_;
  int swsFlags_;
  SwsContext* scaleContext_{nullptr};
};
