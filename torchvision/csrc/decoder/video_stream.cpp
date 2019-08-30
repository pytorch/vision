// Copyright 2004-present Facebook. All Rights Reserved.

#include "video_stream.h"
#include <glog/logging.h>
#include "util.h"

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
} // namespace

VideoStream::VideoStream(
    AVFormatContext* inputCtx,
    int index,
    bool convertPtsToWallTime,
    const VideoFormat& format,
    int64_t loggingUuid)
    : Stream(inputCtx, index, convertPtsToWallTime),
      format_(format),
      loggingUuid_(loggingUuid) {}

VideoStream::~VideoStream() {
  if (sampler_) {
    sampler_->shutdown();
    sampler_.reset();
  }
}

void VideoStream::ensureSampler() {
  if (!sampler_) {
    sampler_ = std::make_unique<VideoSampler>(SWS_AREA, loggingUuid_);
  }
}

int VideoStream::initFormat() {
  // set output format
  if (!Util::validateVideoFormat(format_)) {
    LOG(CRITICAL) << "Invalid video format"
                  << ", width: " << format_.width
                  << ", height: " << format_.height
                  << ", format: " << format_.format
                  << ", minDimension: " << format_.minDimension
                  << ", crop: " << format_.cropImage;
    return -1;
  }

  // keep aspect ratio
  Util::setFormatDimensions(
      format_.width,
      format_.height,
      format_.width,
      format_.height,
      codecCtx_->width,
      codecCtx_->height,
      format_.minDimension,
      0);

  if (format_.format == AV_PIX_FMT_NONE) {
    format_.format = codecCtx_->pix_fmt;
  }
  return format_.width != 0 && format_.height != 0 &&
          format_.format != AV_PIX_FMT_NONE
      ? 0
      : -1;
}

int VideoStream::estimateBytes(bool flush) {
  ensureSampler();
  // check if input format gets changed
  if (!flush && !(sampler_->getInputFormat().video == *frame_)) {
    // - reinit sampler
    SamplerParameters params;
    params.type = MediaType::TYPE_VIDEO;
    params.out.video = format_;
    toVideoFormat(params.in.video, *frame_);
    if (!sampler_->init(params)) {
      return -1;
    }

    VLOG(1) << "Set input video sampler format"
            << ", width: " << params.in.video.width
            << ", height: " << params.in.video.height
            << ", format: " << params.in.video.format
            << " : output video sampler format"
            << ", width: " << format_.width << ", height: " << format_.height
            << ", format: " << format_.format
            << ", minDimension: " << format_.minDimension
            << ", crop: " << format_.cropImage;
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
        codecCtx_->time_base,
        AV_TIME_BASE_Q);
  } else {
    // If the codec time_base is missing then we would've skipped the
    // rescalePackage step to rescale to codec time_base, so here we can
    // rescale straight from the stream time_base into AV_TIME_BASE_Q.
    header->pts = av_rescale_q(
        av_frame_get_best_effort_timestamp(frame_),
        inputCtx_->streams[index_]->time_base,
        AV_TIME_BASE_Q);
  }

  if (convertPtsToWallTime_) {
    keeper_.adjust(header->pts);
  }

  header->keyFrame = frame_->key_frame;
  auto fpsRational = inputCtx_->streams[index_]->avg_frame_rate;
  if (fpsRational.den) {
    header->fps = av_q2d(fpsRational);
  } else {
    header->fps = std::numeric_limits<double>::quiet_NaN();
  }
  header->format.stream = index_;
  header->format.type = TYPE_VIDEO;
  header->format.format.video = format_;
}

} // namespace ffmpeg
