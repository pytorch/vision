// Copyright 2004-present Facebook. All Rights Reserved.

#include "audio_stream.h"
#include <glog/logging.h>
#include <limits>
#include "util.h"

namespace ffmpeg {

namespace {
bool operator==(const AudioFormat& x, const AVCodecContext& y) {
  return x.samples == y.sample_rate && x.channels == y.channels &&
      x.format == y.sample_fmt;
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
    : Stream(inputCtx, index, convertPtsToWallTime), format_(format) {}

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
  if (format_.samples == 0) {
    format_.samples = codecCtx_->sample_rate;
  }
  if (format_.channels == 0) {
    format_.channels = codecCtx_->channels;
  }
  if (format_.format == AV_SAMPLE_FMT_NONE) {
    format_.format = codecCtx_->sample_fmt;
  }

  return format_.samples != 0 && format_.channels != 0 &&
          format_.format != AV_SAMPLE_FMT_NONE
      ? 0
      : -1;
}

int AudioStream::estimateBytes(bool flush) {
  ensureSampler();
  if (!(sampler_->getInputFormat().audio == *codecCtx_)) {
    // - reinit sampler
    SamplerParameters params;
    params.type = MediaType::TYPE_AUDIO;
    params.out.audio = format_;
    toAudioFormat(params.in.audio, *codecCtx_);
    if (flush || !sampler_->init(params)) {
      return -1;
    }

    VLOG(1) << "Set input audio sampler format"
            << ", samples: " << params.in.audio.samples
            << ", channels: " << params.in.audio.channels
            << ", format: " << params.in.audio.format
            << " : output audio sampler format"
            << ", samples: " << format_.samples
            << ", channels: " << format_.channels
            << ", format: " << format_.format;
  }
  return sampler_->getSamplesBytes(frame_);
}

int AudioStream::copyFrameBytes(ByteStorage* out, bool flush) {
  ensureSampler();
  return sampler_->sample(flush ? nullptr : frame_, out);
}

void AudioStream::setHeader(DecoderHeader* header) {
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

  header->keyFrame = 1;
  header->fps = std::numeric_limits<double>::quiet_NaN();
  header->format.type = TYPE_AUDIO;
  header->format.stream = index_;
  header->format.format.audio = format_;
}

} // namespace ffmpeg
