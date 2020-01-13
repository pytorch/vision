// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "stream.h"
#include "time_keeper.h"
#include "video_sampler.h"

namespace ffmpeg {

/**
 * Class uses FFMPEG library to decode one video stream.
 */

class VideoStream : public Stream {
 public:
  VideoStream(
      AVFormatContext* inputCtx,
      int index,
      bool convertPtsToWallTime,
      const VideoFormat& format,
      int64_t loggingUuid = 0);
  ~VideoStream() override;

 private:
  int initFormat() override;
  int estimateBytes(bool flush) override;
  int copyFrameBytes(ByteStorage* out, bool flush) override;
  void setHeader(DecoderHeader* header) override;

  void ensureSampler();

 private:
  std::unique_ptr<VideoSampler> sampler_;
  TimeKeeper keeper_;
  int64_t loggingUuid_{0};
};

}
