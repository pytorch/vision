// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <chrono>
#include <stdlib.h>

namespace ffmpeg {

/**
 * Class keeps the track of the decoded timestamps (us) for media streams.
 */

class TimeKeeper {
 public:
  TimeKeeper() = default;

  // adjust provided @timestamp to the corrected value
  // return advised sleep time before next frame processing in (us)
  ssize_t adjust(ssize_t& decoderTimestamp);

 private:
   ssize_t startTime_{0};
   ssize_t streamTimestamp_{0};
};

}
