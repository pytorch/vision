#pragma once

#include "FfmpegHeaders.h"
#include "Interface.h"

/**
 * Class sample data from AVFrame
 */
class FfmpegSampler {
 public:
  virtual ~FfmpegSampler() = default;
  // return 0 on success and negative number on failure
  virtual int init() = 0;
  // sample from the given frame
  virtual std::unique_ptr<DecodedFrame> sample(const AVFrame* frame) = 0;
};
