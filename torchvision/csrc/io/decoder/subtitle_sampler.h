#pragma once

#include "defs.h"

namespace ffmpeg {

/**
 * Class transcode audio frames from one format into another
 */

class SubtitleSampler : public MediaSampler {
 public:
  SubtitleSampler() = default;
  ~SubtitleSampler() override;

  bool init(const SamplerParameters& params) override;
  int sample(const ByteStorage* in, ByteStorage* out) override;
  void shutdown() override;

  // returns number processed/scaling bytes
  int sample(AVSubtitle* sub, ByteStorage* out);

  // helper serialization/deserialization methods
  static void serialize(const AVSubtitle& sub, ByteStorage* out);
  static bool deserialize(const ByteStorage& buf, AVSubtitle* sub);

 private:
  // close resources
  void cleanUp();
};

} // namespace ffmpeg
