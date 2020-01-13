// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "defs.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace ffmpeg {

/**
 * Class transcode audio frames from one format into another
 */

class SubtitleSampler : public MediaSampler {
 public:
  SubtitleSampler() = default;
  ~SubtitleSampler() override;

  bool init(const SamplerParameters& params) override;
  MediaType getMediaType() const override { return MediaType::TYPE_VIDEO; }
  FormatUnion getInputFormat() const override { return params_.in; }
  FormatUnion getOutFormat() const override { return params_.out; }
  int sample(const ByteStorage* in, ByteStorage* out) override;
  void shutdown() override;

  // returns number processed/scaling bytes
  int sample(AVSubtitle* sub, ByteStorage* out);
  int getSamplesBytes(AVSubtitle* sub) const;

  // helper serialization/deserialization methods
  static void serialize(const AVSubtitle& sub, ByteStorage* out);
  static bool deserialize(const ByteStorage& buf, AVSubtitle* sub);

 private:
  // close resources
  void cleanUp();
 private:
  SamplerParameters params_;
};

}
