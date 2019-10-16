#include "FfmpegAudioStream.h"
#include "FfmpegUtil.h"

using namespace std;

namespace {

bool operator==(const AudioFormat& x, const AVCodecContext& y) {
  return x.samples == y.sample_rate && x.channels == y.channels &&
      x.format == y.sample_fmt;
}

AudioFormat& toAudioFormat(
    AudioFormat& audioFormat,
    const AVCodecContext& codecCtx) {
  audioFormat.samples = codecCtx.sample_rate;
  audioFormat.channels = codecCtx.channels;
  audioFormat.format = codecCtx.sample_fmt;

  return audioFormat;
}

} // namespace

FfmpegAudioStream::FfmpegAudioStream(
    AVFormatContext* inputCtx,
    int index,
    enum AVMediaType avMediaType,
    MediaFormat mediaFormat,
    double seekFrameMargin)
    : FfmpegStream(inputCtx, index, avMediaType, seekFrameMargin),
      mediaFormat_(mediaFormat) {}

FfmpegAudioStream::~FfmpegAudioStream() {}

void FfmpegAudioStream::checkStreamDecodeParams() {
  auto timeBase = getTimeBase();
  if (timeBase.first > 0) {
    CHECK_EQ(timeBase.first, inputCtx_->streams[index_]->time_base.num);
    CHECK_EQ(timeBase.second, inputCtx_->streams[index_]->time_base.den);
  }
}

void FfmpegAudioStream::updateStreamDecodeParams() {
  auto timeBase = getTimeBase();
  if (timeBase.first == 0) {
    mediaFormat_.format.audio.timeBaseNum =
        inputCtx_->streams[index_]->time_base.num;
    mediaFormat_.format.audio.timeBaseDen =
        inputCtx_->streams[index_]->time_base.den;
  }
  mediaFormat_.format.audio.duration = inputCtx_->streams[index_]->duration;
}

int FfmpegAudioStream::initFormat() {
  AudioFormat& format = mediaFormat_.format.audio;

  if (format.samples == 0) {
    format.samples = codecCtx_->sample_rate;
  }
  if (format.channels == 0) {
    format.channels = codecCtx_->channels;
  }
  if (format.format == AV_SAMPLE_FMT_NONE) {
    format.format = codecCtx_->sample_fmt;
    VLOG(2) << "set stream format sample_fmt: " << format.format;
  }

  checkStreamDecodeParams();

  updateStreamDecodeParams();

  if (format.samples > 0 && format.channels > 0 &&
      format.format != AV_SAMPLE_FMT_NONE) {
    return 0;
  } else {
    return -1;
  }
}

unique_ptr<DecodedFrame> FfmpegAudioStream::sampleFrameData() {
  AudioFormat& audioFormat = mediaFormat_.format.audio;

  if (!sampler_ || !(sampler_->getInFormat() == *codecCtx_)) {
    AudioFormat newInFormat;
    newInFormat = toAudioFormat(newInFormat, *codecCtx_);
    sampler_ = make_unique<FfmpegAudioSampler>(newInFormat, audioFormat);
    VLOG(1) << "Set sampler input audio format"
            << ", samples: " << newInFormat.samples
            << ", channels: " << newInFormat.channels
            << ", format: " << newInFormat.format
            << " : output audio sampler format"
            << ", samples: " << audioFormat.samples
            << ", channels: " << audioFormat.channels
            << ", format: " << audioFormat.format;
    int ret = sampler_->init();
    if (ret < 0) {
      VLOG(1) << "Fail to initialize audio sampler";
      return nullptr;
    }
  }
  return sampler_->sample(frame_);
}
