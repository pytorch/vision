#pragma once

#include <atomic>
#include "defs.h"
#include "time_keeper.h"

namespace ffmpeg {

/**
 * Class uses FFMPEG library to decode one media stream (audio or video).
 */

class Stream {
 public:
  Stream(
      AVFormatContext* inputCtx,
      MediaFormat format,
      bool convertPtsToWallTime,
      int64_t loggingUuid);
  virtual ~Stream();

  // returns 0 - on success or negative error
  int openCodec(std::vector<DecoderMetadata>* metadata);
  // returns 1 - if packet got consumed, 0 - if it's not, and < 0 on error
  int decodePacket(
      const AVPacket* packet,
      DecoderOutputMessage* out,
      bool headerOnly,
      bool* hasMsg);
  // returns stream index
  int getIndex() const {
    return format_.stream;
  }
  // returns 1 - if message got a payload, 0 - if it's not, and < 0 on error
  int flush(DecoderOutputMessage* out, bool headerOnly);
  // return media format
  MediaFormat getMediaFormat() const {
    return format_;
  }

 protected:
  virtual int initFormat() = 0;
  // returns number processed bytes from packet, or negative error
  virtual int analyzePacket(const AVPacket* packet, bool* gotFrame);
  // returns number processed bytes from packet, or negative error
  virtual int copyFrameBytes(ByteStorage* out, bool flush) = 0;
  // sets output format
  virtual void setHeader(DecoderHeader* header, bool flush);
  // set frame pts
  virtual void setFramePts(DecoderHeader* header, bool flush);
  // finds codec
  virtual AVCodec* findCodec(AVCodecParameters* params);

 private:
  // returns 1 - if message got a payload, 0 - if it's not, and < 0 on error
  int getMessage(DecoderOutputMessage* out, bool flush, bool headerOnly);

 protected:
  AVFormatContext* const inputCtx_;
  MediaFormat format_;
  const bool convertPtsToWallTime_;
  int64_t loggingUuid_;

  AVCodecContext* codecCtx_{nullptr};
  AVFrame* frame_{nullptr};

  std::atomic<size_t> numGenerator_{0};
  TimeKeeper keeper_;
  // estimated next frame pts for flushing the last frame
  int64_t nextPts_{0};
  double fps_{30.};
};

} // namespace ffmpeg
