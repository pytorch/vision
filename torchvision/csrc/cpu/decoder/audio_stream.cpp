#include "audio_stream.h"
#include <c10/util/Logging.h>
#include <limits>
#include "util.h"

namespace ffmpeg {

namespace {
bool operator==(const AudioFormat& x, const AVFrame& y) {
  return x.samples == y.sample_rate && x.channels == y.channels &&
      x.format == y.format;
}

bool operator==(const AudioFormat& x, const AVCodecContext& y) {
  return x.samples == y.sample_rate && x.channels == y.channels &&
      x.format == y.sample_fmt;
}

AudioFormat& toAudioFormat(AudioFormat& x, const AVFrame& y) {
  x.samples = y.sample_rate;
  x.channels = y.channels;
  x.format = y.format;
  return x;
}

AudioFormat& toAudioFormat(AudioFormat& x, const AVCodecContext& y) {
  x.samples = y.sample_rate;
  x.channels = y.channels;
  x.format = y.sample_fmt;
  return x;
}
} // namespace

AudioStream::AudioStream(
    AVFormatContext* inputCtx,
    int index,
    bool convertPtsToWallTime,
    const AudioFormat& format)
    : Stream(
          inputCtx,
          MediaFormat::makeMediaFormat(format, index),
          convertPtsToWallTime,
          0) {}

AudioStream::~AudioStream() {
  if (sampler_) {
    sampler_->shutdown();
    sampler_.reset();
  }
}

void AudioStream::ensureSampler() {
  if (!sampler_) {
    sampler_ = std::make_unique<AudioSampler>(codecCtx_);
  }
}

int AudioStream::initFormat() {
  // set output format
  if (format_.format.audio.samples == 0) {
    format_.format.audio.samples = codecCtx_->sample_rate;
  }
  if (format_.format.audio.channels == 0) {
    format_.format.audio.channels = codecCtx_->channels;
  }
  if (format_.format.audio.format == AV_SAMPLE_FMT_NONE) {
    format_.format.audio.format = codecCtx_->sample_fmt;
  }

  return format_.format.audio.samples != 0 &&
          format_.format.audio.channels != 0 &&
          format_.format.audio.format != AV_SAMPLE_FMT_NONE
      ? 0
      : -1;
}

int AudioStream::estimateBytes(bool flush) {
  ensureSampler();
  // check if input format gets changed
  if (flush ? !(sampler_->getInputFormat().audio == *codecCtx_)
            : !(sampler_->getInputFormat().audio == *frame_)) {
    // - reinit sampler
    SamplerParameters params;
    params.type = format_.type;
    params.out = format_.format;
    flush ? toAudioFormat(params.in.audio, *codecCtx_)
          : toAudioFormat(params.in.audio, *frame_);
    if (flush || !sampler_->init(params)) {
      return -1;
    }

    VLOG(1) << "Set input audio sampler format"
            << ", samples: " << params.in.audio.samples
            << ", channels: " << params.in.audio.channels
            << ", format: " << params.in.audio.format
            << " : output audio sampler format"
            << ", samples: " << format_.format.audio.samples
            << ", channels: " << format_.format.audio.channels
            << ", format: " << format_.format.audio.format;
  }
  return sampler_->getSamplesBytes(flush ? nullptr : frame_);
}

int AudioStream::copyFrameBytes(ByteStorage* out, bool flush) {
  ensureSampler();
  return sampler_->sample(flush ? nullptr : frame_, out);
}

} // namespace ffmpeg
