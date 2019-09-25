#include "FfmpegVideoSampler.h"
#include "FfmpegUtil.h"

using namespace std;

FfmpegVideoSampler::FfmpegVideoSampler(
    const VideoFormat& in,
    const VideoFormat& out,
    int swsFlags)
    : inFormat_(in), outFormat_(out), swsFlags_(swsFlags) {}

FfmpegVideoSampler::~FfmpegVideoSampler() {
  if (scaleContext_) {
    sws_freeContext(scaleContext_);
    scaleContext_ = nullptr;
  }
}

int FfmpegVideoSampler::init() {
  VLOG(1) << "Input format: width " << inFormat_.width << ", height "
          << inFormat_.height << ", format " << inFormat_.format
          << ", minDimension " << inFormat_.minDimension;
  VLOG(1) << "Scale format: width " << outFormat_.width << ", height "
          << outFormat_.height << ", format " << outFormat_.format
          << ", minDimension " << outFormat_.minDimension;

  scaleContext_ = sws_getContext(
      inFormat_.width,
      inFormat_.height,
      (AVPixelFormat)inFormat_.format,
      outFormat_.width,
      outFormat_.height,
      static_cast<AVPixelFormat>(outFormat_.format),
      swsFlags_,
      nullptr,
      nullptr,
      nullptr);
  if (scaleContext_) {
    return 0;
  } else {
    return -1;
  }
}

int32_t FfmpegVideoSampler::getImageBytes() const {
  return av_image_get_buffer_size(
      (AVPixelFormat)outFormat_.format, outFormat_.width, outFormat_.height, 1);
}

// https://ffmpeg.org/doxygen/3.4/scaling_video_8c-example.html#a10
unique_ptr<DecodedFrame> FfmpegVideoSampler::sample(const AVFrame* frame) {
  if (!frame) {
    return nullptr; // no flush for videos
  }
  // scaled and cropped image
  auto outImageSize = getImageBytes();
  AvDataPtr frameData(static_cast<uint8_t*>(av_malloc(outImageSize)));

  uint8_t* scalePlanes[4] = {nullptr};
  int scaleLines[4] = {0};

  int result;
  if ((result = av_image_fill_arrays(
           scalePlanes,
           scaleLines,
           frameData.get(),
           static_cast<AVPixelFormat>(outFormat_.format),
           outFormat_.width,
           outFormat_.height,
           1)) < 0) {
    LOG(ERROR) << "av_image_fill_arrays failed, err: "
               << ffmpeg_util::getErrorDesc(result);
    return nullptr;
  }

  if ((result = sws_scale(
           scaleContext_,
           frame->data,
           frame->linesize,
           0,
           inFormat_.height,
           scalePlanes,
           scaleLines)) < 0) {
    LOG(ERROR) << "sws_scale failed, err: "
               << ffmpeg_util::getErrorDesc(result);
    return nullptr;
  }

  return make_unique<DecodedFrame>(std::move(frameData), outImageSize, 0);
}
