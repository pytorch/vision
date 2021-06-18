#pragma once

#include "stream.h"
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
      int64_t loggingUuid);
  ~VideoStream() override;

 private:
  int initFormat() override;
  int copyFrameBytes(ByteStorage* out, bool flush) override;
  void setHeader(DecoderHeader* header, bool flush) override;

 private:
  std::unique_ptr<VideoSampler> sampler_;
};

} // namespace ffmpeg
