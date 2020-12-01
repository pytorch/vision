#include "subtitle_sampler.h"
#include <c10/util/Logging.h>
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

int SubtitleSampler::sample(AVSubtitle* sub, ByteStorage* out) {
  if (!sub || !out) {
    return 0; // flush
  }

  out->ensure(Util::size(*sub));

  return Util::serialize(*sub, out);
}

int SubtitleSampler::sample(const ByteStorage* in, ByteStorage* out) {
  if (in && out) {
    // Get a writable copy
    if (size_t len = in->length()) {
      out->ensure(len);
      memcpy(out->writableTail(), in->data(), len);
    }
    return out->length();
  }
  return 0;
}

void SubtitleSampler::cleanUp() {}

} // namespace ffmpeg
