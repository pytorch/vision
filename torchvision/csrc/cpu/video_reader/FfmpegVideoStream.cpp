#include "FfmpegVideoStream.h"
#include "FfmpegUtil.h"

using namespace std;

namespace {

bool operator==(const VideoFormat& x, const AVFrame& y) {
  return x.width == y.width && x.height == y.height &&
      x.format == static_cast<AVPixelFormat>(y.format);
}

VideoFormat toVideoFormat(const AVFrame& frame) {
  VideoFormat videoFormat;
  videoFormat.width = frame.width;
  videoFormat.height = frame.height;
  videoFormat.format = static_cast<AVPixelFormat>(frame.format);

  return videoFormat;
}

} // namespace

FfmpegVideoStream::FfmpegVideoStream(
    AVFormatContext* inputCtx,
    int index,
    enum AVMediaType avMediaType,
    MediaFormat mediaFormat,
    double seekFrameMargin)
    : FfmpegStream(inputCtx, index, avMediaType, seekFrameMargin),
      mediaFormat_(mediaFormat) {}

FfmpegVideoStream::~FfmpegVideoStream() {}

void FfmpegVideoStream::checkStreamDecodeParams() {
  auto timeBase = getTimeBase();
  if (timeBase.first > 0) {
    CHECK_EQ(timeBase.first, inputCtx_->streams[index_]->time_base.num);
    CHECK_EQ(timeBase.second, inputCtx_->streams[index_]->time_base.den);
  }
}

void FfmpegVideoStream::updateStreamDecodeParams() {
  auto timeBase = getTimeBase();
  if (timeBase.first == 0) {
    mediaFormat_.format.video.timeBaseNum =
        inputCtx_->streams[index_]->time_base.num;
    mediaFormat_.format.video.timeBaseDen =
        inputCtx_->streams[index_]->time_base.den;
  }
  mediaFormat_.format.video.duration = inputCtx_->streams[index_]->duration;
}

int FfmpegVideoStream::initFormat() {
  // set output format
  VideoFormat& format = mediaFormat_.format.video;
  if (!ffmpeg_util::validateVideoFormat(format)) {
    LOG(ERROR) << "Invalid video format";
    return -1;
  }

  format.fps = av_q2d(
      av_guess_frame_rate(inputCtx_, inputCtx_->streams[index_], nullptr));

  // keep aspect ratio
  ffmpeg_util::setFormatDimensions(
      format.width,
      format.height,
      format.width,
      format.height,
      codecCtx_->width,
      codecCtx_->height,
      format.minDimension);

  VLOG(1) << "After adjusting, video format"
          << ", width: " << format.width << ", height: " << format.height
          << ", format: " << format.format
          << ", minDimension: " << format.minDimension;

  if (format.format == AV_PIX_FMT_NONE) {
    format.format = codecCtx_->pix_fmt;
    VLOG(1) << "Set pixel format: " << format.format;
  }

  checkStreamDecodeParams();

  updateStreamDecodeParams();

  return format.width != 0 && format.height != 0 &&
          format.format != AV_PIX_FMT_NONE
      ? 0
      : -1;
}

unique_ptr<DecodedFrame> FfmpegVideoStream::sampleFrameData() {
  VideoFormat& format = mediaFormat_.format.video;
  if (!sampler_ || !(sampler_->getInFormat() == *frame_)) {
    VideoFormat newInFormat = toVideoFormat(*frame_);
    sampler_ = make_unique<FfmpegVideoSampler>(newInFormat, format, SWS_AREA);
    VLOG(1) << "Set input video sampler format"
            << ", width: " << newInFormat.width
            << ", height: " << newInFormat.height
            << ", format: " << newInFormat.format
            << " : output video sampler format"
            << ", width: " << format.width << ", height: " << format.height
            << ", format: " << format.format
            << ", minDimension: " << format.minDimension;
    int ret = sampler_->init();
    if (ret < 0) {
      VLOG(1) << "Fail to initialize video sampler";
      return nullptr;
    }
  }
  return sampler_->sample(frame_);
}
