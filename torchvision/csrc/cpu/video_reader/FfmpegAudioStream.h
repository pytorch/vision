#pragma once

#include <utility>
#include "FfmpegAudioSampler.h"
#include "FfmpegStream.h"

/**
 * Class uses FFMPEG library to decode one video stream.
 */
class FfmpegAudioStream : public FfmpegStream {
 public:
  explicit FfmpegAudioStream(
      AVFormatContext* inputCtx,
      int index,
      enum AVMediaType avMediaType,
      MediaFormat mediaFormat,
      double seekFrameMargin);

  ~FfmpegAudioStream() override;

  // FfmpegStream overrides
  MediaType getMediaType() const override {
    return MediaType::TYPE_AUDIO;
  }

  FormatUnion getMediaFormat() const override {
    return mediaFormat_.format;
  }

  int64_t getStartPts() const override {
    return mediaFormat_.format.audio.startPts;
  }
  int64_t getEndPts() const override {
    return mediaFormat_.format.audio.endPts;
  }
  // return numerator and denominator of time base
  std::pair<int, int> getTimeBase() const {
    return std::make_pair(
        mediaFormat_.format.audio.timeBaseNum,
        mediaFormat_.format.audio.timeBaseDen);
  }

  void checkStreamDecodeParams();

  void updateStreamDecodeParams();

 protected:
  int initFormat() override;
  std::unique_ptr<DecodedFrame> sampleFrameData() override;

 private:
  MediaFormat mediaFormat_;
  std::unique_ptr<FfmpegAudioSampler> sampler_{nullptr};
};
