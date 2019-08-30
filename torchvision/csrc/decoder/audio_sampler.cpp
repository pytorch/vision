// Copyright 2004-present Facebook. All Rights Reserved.

#include "audio_sampler.h"
#include <glog/logging.h>
#include "util.h"

// www.ffmpeg.org/doxygen/1.1/doc_2examples_2resampling_audio_8c-example.html#a24

#ifndef SWR_CH_MAX
#define SWR_CH_MAX 32
#endif

namespace ffmpeg {

namespace {
int preparePlanes(
    const AudioFormat& fmt,
    const uint8_t* buffer,
    int numSamples,
    uint8_t** planes) {
  int result;
  if ((result = av_samples_fill_arrays(
           planes,
           nullptr, // linesize is not needed
           buffer,
           fmt.channels,
           numSamples,
           (AVSampleFormat)fmt.format,
           1)) < 0) {
    LOG(CRITICAL) << "av_samples_fill_arrays failed, err: "
                  << Util::generateErrorDesc(result)
                  << ", numSamples: " << numSamples << ", fmt: " << fmt.format;
  }
  return result;
}
} // namespace

AudioSampler::AudioSampler(void* logCtx) : logCtx_(logCtx) {}

AudioSampler::~AudioSampler() {
  cleanUp();
}

void AudioSampler::shutdown() {
  cleanUp();
}

bool AudioSampler::init(const SamplerParameters& params) {
  cleanUp();

  if (params.type != MediaType::TYPE_AUDIO) {
    LOG(CRITICAL) << "Invalid media type, expected MediaType::TYPE_AUDIO";
    return false;
  }

  swrContext_ = swr_alloc_set_opts(
      nullptr,
      av_get_default_channel_layout(params.out.audio.channels),
      (AVSampleFormat)params.out.audio.format,
      params.out.audio.samples,
      av_get_default_channel_layout(params.in.audio.channels),
      (AVSampleFormat)params.in.audio.format,
      params.in.audio.samples,
      0,
      logCtx_);
  if (swrContext_ == nullptr) {
    LOG(CRITICAL) << "Cannot allocate SwrContext";
    return false;
  }

  int result;
  if ((result = swr_init(swrContext_)) < 0) {
    LOG(CRITICAL) << "swr_init faield, err: " << Util::generateErrorDesc(result)
                  << ", in -> format: " << params.in.audio.format
                  << ", channels: " << params.in.audio.channels
                  << ", samples: " << params.in.audio.samples
                  << ", out -> format: " << params.out.audio.format
                  << ", channels: " << params.out.audio.channels
                  << ", samples: " << params.out.audio.samples;
    return false;
  }

  // set formats
  params_ = params;
  return true;
}

int AudioSampler::numOutputSamples(int inSamples) const {
  return av_rescale_rnd(
      swr_get_delay(swrContext_, params_.in.audio.samples) + inSamples,
      params_.out.audio.samples,
      params_.in.audio.samples,
      AV_ROUND_UP);
}

int AudioSampler::getSamplesBytes(AVFrame* frame) const {
  return av_get_bytes_per_sample((AVSampleFormat)params_.out.audio.format) *
      numOutputSamples(frame ? frame->nb_samples : 0) *
      params_.out.audio.channels;
}

int AudioSampler::sample(
    const uint8_t* inPlanes[],
    int inNumSamples,
    ByteStorage* out,
    int outNumSamples) {
  uint8_t* outPlanes[SWR_CH_MAX] = {nullptr};
  int result;
  if ((result = preparePlanes(
           params_.out.audio, out->writableTail(), outNumSamples, outPlanes)) <
      0) {
    return result;
  }

  if ((result = swr_convert(
           swrContext_, &outPlanes[0], outNumSamples, inPlanes, inNumSamples)) <
      0) {
    LOG(CRITICAL) << "swr_convert faield, err: "
                  << Util::generateErrorDesc(result);
    return result;
  }

  CHECK_LE(result, outNumSamples);

  if ((result = av_samples_get_buffer_size(
           nullptr,
           params_.out.audio.channels,
           result,
           (AVSampleFormat)params_.out.audio.format,
           1)) > 0) {
    out->append(result);
  }
  return result;
}

int AudioSampler::sample(AVFrame* frame, ByteStorage* out) {
  const auto outNumSamples = numOutputSamples(frame ? frame->nb_samples : 0);

  if (!outNumSamples) {
    return 0;
  }

  const auto samplesBytes =
      av_get_bytes_per_sample((AVSampleFormat)params_.out.audio.format) *
      outNumSamples * params_.out.audio.channels;

  // bytes must be allocated
  CHECK_LE(samplesBytes, out->tail());

  return sample(
      frame ? (const uint8_t**)&frame->data[0] : nullptr,
      frame ? frame->nb_samples : 0,
      out,
      outNumSamples);
}

int AudioSampler::sample(const ByteStorage* in, ByteStorage* out) {
  const auto inSampleSize =
      av_get_bytes_per_sample((AVSampleFormat)params_.in.audio.format);

  const auto inNumSamples =
      !in ? 0 : in->length() / inSampleSize / params_.in.audio.channels;

  const auto outNumSamples = numOutputSamples(inNumSamples);

  if (!outNumSamples) {
    return 0;
  }

  const auto samplesBytes =
      av_get_bytes_per_sample((AVSampleFormat)params_.out.audio.format) *
      outNumSamples * params_.out.audio.channels;

  out->clear();
  out->ensure(samplesBytes);

  uint8_t* inPlanes[SWR_CH_MAX] = {nullptr};
  int result;
  if (in &&
      (result = preparePlanes(
           params_.in.audio, in->data(), inNumSamples, inPlanes)) < 0) {
    return result;
  }

  return sample(
      in ? (const uint8_t**)inPlanes : nullptr,
      inNumSamples,
      out,
      outNumSamples);
}

void AudioSampler::cleanUp() {
  if (swrContext_) {
    swr_free(&swrContext_);
    swrContext_ = nullptr;
  }
}

} // namespace ffmpeg
