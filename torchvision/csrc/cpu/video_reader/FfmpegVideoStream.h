#pragma once

#include <utility>
#include "FfmpegStream.h"
#include "FfmpegVideoSampler.h"

/**
 * Class uses FFMPEG library to decode one video stream.
 */
class FfmpegVideoStream : public FfmpegStream {
 public:
  explicit FfmpegVideoStream(
      AVFormatContext* inputCtx,
      int index,
      enum AVMediaType avMediaType,
      MediaFormat mediaFormat,
      double seekFrameMargin);

  ~FfmpegVideoStream() override;

  // FfmpegStream overrides
  MediaType getMediaType() const override {
    return MediaType::TYPE_VIDEO;
  }

  FormatUnion getMediaFormat() const override {
    return mediaFormat_.format;
  }

  int64_t getStartPts() const override {
    return mediaFormat_.format.video.startPts;
  }
  int64_t getEndPts() const override {
    return mediaFormat_.format.video.endPts;
  }
  // return numerator and denominator of time base
  std::pair<int, int> getTimeBase() const {
    return std::make_pair(
        mediaFormat_.format.video.timeBaseNum,
        mediaFormat_.format.video.timeBaseDen);
  }

  void checkStreamDecodeParams();

  void updateStreamDecodeParams();

 protected:
  int initFormat() override;
  std::unique_ptr<DecodedFrame> sampleFrameData() override;

 private:
  MediaFormat mediaFormat_;
  std::unique_ptr<FfmpegVideoSampler> sampler_{nullptr};
};
