#include "FfmpegAudioSampler.h"
#include <memory>
#include "FfmpegUtil.h"

using namespace std;

FfmpegAudioSampler::FfmpegAudioSampler(
    const AudioFormat& in,
    const AudioFormat& out)
    : inFormat_(in), outFormat_(out) {}

FfmpegAudioSampler::~FfmpegAudioSampler() {
  if (swrContext_) {
    swr_free(&swrContext_);
  }
}

int FfmpegAudioSampler::init() {
  swrContext_ = swr_alloc_set_opts(
      nullptr, // we're allocating a new context
      av_get_default_channel_layout(outFormat_.channels), // out_ch_layout
      static_cast<AVSampleFormat>(outFormat_.format), // out_sample_fmt
      outFormat_.samples, // out_sample_rate
      av_get_default_channel_layout(inFormat_.channels), // in_ch_layout
      static_cast<AVSampleFormat>(inFormat_.format), // in_sample_fmt
      inFormat_.samples, // in_sample_rate
      0, // log_offset
      nullptr); // log_ctx
  if (swrContext_ == nullptr) {
    LOG(ERROR) << "swr_alloc_set_opts fails";
    return -1;
  }
  int result = 0;
  if ((result = swr_init(swrContext_)) < 0) {
    LOG(ERROR) << "swr_init failed, err: " << ffmpeg_util::getErrorDesc(result)
               << ", in -> format: " << inFormat_.format
               << ", channels: " << inFormat_.channels
               << ", samples: " << inFormat_.samples
               << ", out -> format: " << outFormat_.format
               << ", channels: " << outFormat_.channels
               << ", samples: " << outFormat_.samples;
    return -1;
  }
  return 0;
}

int64_t FfmpegAudioSampler::getSampleBytes(const AVFrame* frame) const {
  auto outSamples = getOutNumSamples(frame->nb_samples);

  return av_samples_get_buffer_size(
      nullptr,
      outFormat_.channels,
      outSamples,
      static_cast<AVSampleFormat>(outFormat_.format),
      1);
}

// https://www.ffmpeg.org/doxygen/3.2/group__lswr.html
unique_ptr<DecodedFrame> FfmpegAudioSampler::sample(const AVFrame* frame) {
  if (!frame) {
    return nullptr; // no flush for videos
  }

  auto inNumSamples = frame->nb_samples;
  auto outNumSamples = getOutNumSamples(frame->nb_samples);

  auto outSampleSize = getSampleBytes(frame);
  AvDataPtr frameData(static_cast<uint8_t*>(av_malloc(outSampleSize)));

  uint8_t* outPlanes[AVRESAMPLE_MAX_CHANNELS];
  int result = 0;
  if ((result = av_samples_fill_arrays(
           outPlanes,
           nullptr, // linesize is not needed
           frameData.get(),
           outFormat_.channels,
           outNumSamples,
           static_cast<AVSampleFormat>(outFormat_.format),
           1)) < 0) {
    LOG(ERROR) << "av_samples_fill_arrays failed, err: "
               << ffmpeg_util::getErrorDesc(result)
               << ", outNumSamples: " << outNumSamples
               << ", format: " << outFormat_.format;
    return nullptr;
  }

  if ((result = swr_convert(
           swrContext_,
           &outPlanes[0],
           outNumSamples,
           (const uint8_t**)&frame->data[0],
           inNumSamples)) < 0) {
    LOG(ERROR) << "swr_convert faield, err: "
               << ffmpeg_util::getErrorDesc(result);
    return nullptr;
  }
  // result returned by swr_convert is the No. of actual output samples.
  // So update the buffer size using av_samples_get_buffer_size
  result = av_samples_get_buffer_size(
      nullptr,
      outFormat_.channels,
      result,
      static_cast<AVSampleFormat>(outFormat_.format),
      1);

  return make_unique<DecodedFrame>(std::move(frameData), result, 0);
}
/*
Because of decoding delay, the returned value is an upper bound of No. of
output samples
*/
int64_t FfmpegAudioSampler::getOutNumSamples(int inNumSamples) const {
  return av_rescale_rnd(
      swr_get_delay(swrContext_, inFormat_.samples) + inNumSamples,
      outFormat_.samples,
      inFormat_.samples,
      AV_ROUND_UP);
}
