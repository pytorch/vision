// Copyright 2004-present Facebook. All Rights Reserved.

#include "subtitle_sampler.h"
#include "util.h"

namespace ffmpeg {

SubtitleSampler::~SubtitleSampler() {
  cleanUp();
}

void SubtitleSampler::shutdown() {
  cleanUp();
}

bool SubtitleSampler::init(const SamplerParameters& params) {
  cleanUp();
  // set formats
  params_ = params;
  return true;
}

int SubtitleSampler::getSamplesBytes(AVSubtitle* sub) const {
  return Util::size(*sub);
}

int SubtitleSampler::sample(AVSubtitle* sub, ByteStorage* out) {
  if (!sub) {
    return 0; // flush
  }

  return Util::serialize(*sub, out);
}

int SubtitleSampler::sample(const ByteStorage* in, ByteStorage* out) {
  if (in) {
    // Get a writable copy
    *out = *in;
    return out->length();
  }
  return 0;
}

void SubtitleSampler::cleanUp() {
}

}
