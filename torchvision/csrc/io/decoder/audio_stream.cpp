#include "audio_stream.h"
#include <c10/util/Logging.h>
#include <limits>
#include "util.h"

namespace ffmpeg {

namespace {
bool operator==(const AudioFormat& x, const AVFrame& y) {
  return x.samples == static_cast<size_t>(y.sample_rate) &&
      x.channels == static_cast<size_t>(y.channels) && x.format == y.format;
}

bool operator==(const AudioFormat& x, const AVCodecContext& y) {
  return x.samples == static_cast<size_t>(y.sample_rate) &&
      x.channels == static_cast<size_t>(y.channels) && x.format == y.sample_fmt;
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

// copies audio sample bytes via swr_convert call in audio_sampler.cpp
int AudioStream::copyFrameBytes(ByteStorage* out, bool flush) {
  if (!sampler_) {
    sampler_ = std::make_unique<AudioSampler>(codecCtx_);
  }
  // check if input format gets changed
  if (flush ? !(sampler_->getInputFormat().audio == *codecCtx_)
            : !(sampler_->getInputFormat().audio == *frame_)) {
    // - reinit sampler
    SamplerParameters params;
    params.type = format_.type;
    params.out = format_.format;
    params.in = FormatUnion();
    flush ? toAudioFormat(params.in.audio, *codecCtx_)
          : toAudioFormat(params.in.audio, *frame_);
    if (!sampler_->init(params)) {
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
  // calls to a sampler that converts the audio samples and copies them to the
  // out buffer via ffmpeg::swr_convert
  return sampler_->sample(flush ? nullptr : frame_, out);
}

} // namespace ffmpeg
