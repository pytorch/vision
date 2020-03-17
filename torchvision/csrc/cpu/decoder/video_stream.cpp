#include "video_stream.h"
#include <c10/util/Logging.h>
#include "util.h"

namespace ffmpeg {

namespace {
bool operator==(const VideoFormat& x, const AVFrame& y) {
  return x.width == y.width && x.height == y.height && x.format == y.format;
}

bool operator==(const VideoFormat& x, const AVCodecContext& y) {
  return x.width == y.width && x.height == y.height && x.format == y.pix_fmt;
}

VideoFormat& toVideoFormat(VideoFormat& x, const AVFrame& y) {
  x.width = y.width;
  x.height = y.height;
  x.format = y.format;
  return x;
}

VideoFormat& toVideoFormat(VideoFormat& x, const AVCodecContext& y) {
  x.width = y.width;
  x.height = y.height;
  x.format = y.pix_fmt;
  return x;
}
} // namespace

VideoStream::VideoStream(
    AVFormatContext* inputCtx,
    int index,
    bool convertPtsToWallTime,
    const VideoFormat& format,
    int64_t loggingUuid)
    : Stream(
          inputCtx,
          MediaFormat::makeMediaFormat(format, index),
          convertPtsToWallTime,
          loggingUuid) {}

VideoStream::~VideoStream() {
  if (sampler_) {
    sampler_->shutdown();
    sampler_.reset();
  }
}

int VideoStream::initFormat() {
  // set output format
  if (!Util::validateVideoFormat(format_.format.video)) {
    LOG(ERROR) << "Invalid video format"
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
      format_.format.video.maxDimension,
      0);

  if (format_.format.video.format == AV_PIX_FMT_NONE) {
    format_.format.video.format = codecCtx_->pix_fmt;
  }
  return format_.format.video.width != 0 && format_.format.video.height != 0 &&
          format_.format.video.format != AV_PIX_FMT_NONE
      ? 0
      : -1;
}

int VideoStream::copyFrameBytes(ByteStorage* out, bool flush) {
  if (!sampler_) {
    sampler_ = std::make_unique<VideoSampler>(SWS_AREA, loggingUuid_);
  }

  // check if input format gets changed
  if (flush ? !(sampler_->getInputFormat().video == *codecCtx_)
            : !(sampler_->getInputFormat().video == *frame_)) {
    // - reinit sampler
    SamplerParameters params;
    params.type = format_.type;
    params.out = format_.format;
    params.in = FormatUnion(0);
    flush ? toVideoFormat(params.in.video, *codecCtx_)
          : toVideoFormat(params.in.video, *frame_);
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

  return sampler_->sample(flush ? nullptr : frame_, out);
}

void VideoStream::setHeader(DecoderHeader* header, bool flush) {
  Stream::setHeader(header, flush);
  if (!flush) { // no frames for video flush
    header->keyFrame = frame_->key_frame;
    header->fps = av_q2d(av_guess_frame_rate(
        inputCtx_, inputCtx_->streams[format_.stream], nullptr));
  }
}

} // namespace ffmpeg
