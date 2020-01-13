// Copyright 2004-present Facebook. All Rights Reserved.

#include "video_stream.h"
#include "util.h"
#include <glog/logging.h>

namespace ffmpeg {

namespace {
bool operator==(const VideoFormat& x, const AVFrame& y) {
  return x.width == y.width && x.height == y.height && x.format == y.format;
}

VideoFormat& toVideoFormat(VideoFormat& x, const AVFrame& y) {
  x.width = y.width;
  x.height = y.height;
  x.format = y.format;
  return x;
}
}

VideoStream::VideoStream(
    AVFormatContext* inputCtx,
    int index,
    bool convertPtsToWallTime,
    const VideoFormat& format,
    int64_t loggingUuid)
    : Stream(inputCtx,
             MediaFormat::makeMediaFormat(format, index),
             convertPtsToWallTime),
      loggingUuid_(loggingUuid) {}

VideoStream::~VideoStream() {
  if (sampler_) {
    sampler_->shutdown();
    sampler_.reset();
  }
}

void VideoStream::ensureSampler() {
  if (!sampler_) {
    sampler_ =
        std::make_unique<VideoSampler>(SWS_AREA, loggingUuid_);
  }
}

int VideoStream::initFormat() {
  // set output format
  if (!Util::validateVideoFormat(format_.format.video)) {
    LOG(CRITICAL) << "Invalid video format"
                  << ", width: " << format_.format.video.width
                  << ", height: " << format_.format.video.height
                  << ", format: " << format_.format.video.format
                  << ", minDimension: " << format_.format.video.minDimension
                  << ", crop: " << format_.format.video.cropImage;
    return -1;
  }

  // keep aspect ratio
  Util::setFormatDimensions(
      format_.format.video.width,
      format_.format.video.height,
      format_.format.video.width,
      format_.format.video.height,
      codecCtx_->width,
      codecCtx_->height,
      format_.format.video.minDimension,
      0);

  if (format_.format.video.format == AV_PIX_FMT_NONE) {
    format_.format.video.format = codecCtx_->pix_fmt;
  }
  return format_.format.video.width != 0 &&
         format_.format.video.height != 0 &&
         format_.format.video.format != AV_PIX_FMT_NONE
      ? 0
      : -1;
}

int VideoStream::estimateBytes(bool flush) {
  ensureSampler();
  // check if input format gets changed
  if (!flush && !(sampler_->getInputFormat().video == *frame_)) {
    // - reinit sampler
    SamplerParameters params;
    params.type = format_.type;
    params.out = format_.format;
    toVideoFormat(params.in.video, *frame_);
    if (!sampler_->init(params)) {
      return -1;
    }

    VLOG(1) << "Set input video sampler format"
            << ", width: " << params.in.video.width
            << ", height: " << params.in.video.height
            << ", format: " << params.in.video.format
            << " : output video sampler format"
            << ", width: " << format_.format.video.width
            << ", height: " << format_.format.video.height
            << ", format: " << format_.format.video.format
            << ", minDimension: " << format_.format.video.minDimension
            << ", crop: " << format_.format.video.cropImage;
  }
  return sampler_->getImageBytes();
}

int VideoStream::copyFrameBytes(ByteStorage* out, bool flush) {
  ensureSampler();
  return sampler_->sample(flush ? nullptr : frame_, out);
}

void VideoStream::setHeader(DecoderHeader* header) {
  header->seqno = numGenerator_++;

  if (codecCtx_->time_base.num != 0) {
    header->pts = av_rescale_q(
         av_frame_get_best_effort_timestamp(frame_),
         codecCtx_->time_base, AV_TIME_BASE_Q);
  } else {
    // If the codec time_base is missing then we would've skipped the
    // rescalePackage step to rescale to codec time_base, so here we can
    // rescale straight from the stream time_base into AV_TIME_BASE_Q.
    header->pts = av_rescale_q(
         av_frame_get_best_effort_timestamp(frame_),
         inputCtx_->streams[format_.stream]->time_base, AV_TIME_BASE_Q);
  }

  if (convertPtsToWallTime_) {
    keeper_.adjust(header->pts);
  }

  header->keyFrame = frame_->key_frame;
  auto fpsRational = inputCtx_->streams[format_.stream]->avg_frame_rate;
  if (fpsRational.den) {
    header->fps = av_q2d(fpsRational);
  } else {
    header->fps = std::numeric_limits<double>::quiet_NaN();
  }
  header->format = format_;
}

}
