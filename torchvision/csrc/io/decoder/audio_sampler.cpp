#include "audio_sampler.h"
#include <c10/util/Logging.h>
#include "util.h"

#define AVRESAMPLE_MAX_CHANNELS 32

// www.ffmpeg.org/doxygen/1.1/doc_2examples_2resampling_audio_8c-example.html#a24
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
    LOG(ERROR) << "av_samples_fill_arrays failed, err: "
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
    LOG(ERROR) << "Invalid media type, expected MediaType::TYPE_AUDIO";
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
    LOG(ERROR) << "Cannot allocate SwrContext";
    return false;
  }

  int result;
  if ((result = swr_init(swrContext_)) < 0) {
    LOG(ERROR) << "swr_init failed, err: " << Util::generateErrorDesc(result)
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
  return swr_get_out_samples(swrContext_, inSamples);
}

int AudioSampler::sample(
    const uint8_t* inPlanes[],
    int inNumSamples,
    ByteStorage* out,
    int outNumSamples) {
  int result;
  int outBufferBytes = av_samples_get_buffer_size(
      nullptr,
      params_.out.audio.channels,
      outNumSamples,
      (AVSampleFormat)params_.out.audio.format,
      1);

  if (out) {
    out->ensure(outBufferBytes);

    uint8_t* outPlanes[AVRESAMPLE_MAX_CHANNELS] = {nullptr};

    if ((result = preparePlanes(
             params_.out.audio,
             out->writableTail(),
             outNumSamples,
             outPlanes)) < 0) {
      return result;
    }

    if ((result = swr_convert(
             swrContext_,
             &outPlanes[0],
             outNumSamples,
             inPlanes,
             inNumSamples)) < 0) {
      LOG(ERROR) << "swr_convert failed, err: "
                 << Util::generateErrorDesc(result);
      return result;
    }

    TORCH_CHECK_LE(result, outNumSamples);

    if (result) {
      if ((result = av_samples_get_buffer_size(
               nullptr,
               params_.out.audio.channels,
               result,
               (AVSampleFormat)params_.out.audio.format,
               1)) >= 0) {
        out->append(result);
      } else {
        LOG(ERROR) << "av_samples_get_buffer_size failed, err: "
                   << Util::generateErrorDesc(result);
      }
    }
  } else {
    // allocate a temporary buffer
    auto* tmpBuffer = static_cast<uint8_t*>(av_malloc(outBufferBytes));
    if (!tmpBuffer) {
      LOG(ERROR) << "av_alloc failed, for size: " << outBufferBytes;
      return -1;
    }

    uint8_t* outPlanes[AVRESAMPLE_MAX_CHANNELS] = {nullptr};

    if ((result = preparePlanes(
             params_.out.audio, tmpBuffer, outNumSamples, outPlanes)) < 0) {
      av_free(tmpBuffer);
      return result;
    }

    if ((result = swr_convert(
             swrContext_,
             &outPlanes[0],
             outNumSamples,
             inPlanes,
             inNumSamples)) < 0) {
      LOG(ERROR) << "swr_convert failed, err: "
                 << Util::generateErrorDesc(result);
      av_free(tmpBuffer);
      return result;
    }

    av_free(tmpBuffer);

    TORCH_CHECK_LE(result, outNumSamples);

    if (result) {
      result = av_samples_get_buffer_size(
          nullptr,
          params_.out.audio.channels,
          result,
          (AVSampleFormat)params_.out.audio.format,
          1);
    }
  }

  return result;
}

int AudioSampler::sample(AVFrame* frame, ByteStorage* out) {
  const auto outNumSamples = numOutputSamples(frame ? frame->nb_samples : 0);

  if (!outNumSamples) {
    return 0;
  }

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

  uint8_t* inPlanes[AVRESAMPLE_MAX_CHANNELS] = {nullptr};
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
