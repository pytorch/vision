// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include "FfmpegHeaders.h"
#include "Interface.h"

/*
Class uses FFMPEG library to decode one media stream (audio or video).
*/
class FfmpegStream {
 public:
  FfmpegStream(
      AVFormatContext* inputCtx,
      int index,
      enum AVMediaType avMediaType,
      double seekFrameMargin);
  virtual ~FfmpegStream();

  // returns 0 - on success or negative error
  int openCodecContext();
  // returns stream index
  int getIndex() const {
    return index_;
  }
  // returns number decoded/sampled bytes
  std::unique_ptr<DecodedFrame> getFrameData(int getPtsOnly);
  // flush the stream at the end of decoding.
  // Return 0 on success and -1 when cache is drained
  void flush(int getPtsOnly, DecoderOutput& decoderOutput);
  // seek a frame
  int seekFrame(int64_t ts);
  // send an AVPacket
  int sendPacket(const AVPacket* packet);
  // receive AVFrame
  int receiveFrame();
  // receive all available frames from the internal buffer
  void receiveAvailFrames(int getPtsOnly, DecoderOutput& decoderOutput);
  // return media type
  virtual MediaType getMediaType() const = 0;
  // return media format
  virtual FormatUnion getMediaFormat() const = 0;
  // return start presentation timestamp
  virtual int64_t getStartPts() const = 0;
  // return end presentation timestamp
  virtual int64_t getEndPts() const = 0;
  // is the pts of most recent frame within range?
  bool isFramePtsInRange();
  // does the pts of most recent frame exceed range?
  bool isFramePtsExceedRange();

 protected:
  virtual int initFormat() = 0;
  // returns a decoded frame
  virtual std::unique_ptr<DecodedFrame> sampleFrameData() = 0;

 protected:
  AVFormatContext* const inputCtx_;
  const int index_;
  enum AVMediaType avMediaType_;

  AVCodecContext* codecCtx_{nullptr};
  AVFrame* frame_{nullptr};
  // pts of last decoded frame
  int64_t last_pts_{0};
  double seekFrameMargin_{1.0};
};
