#pragma once

#include <stdlib.h>
#include <chrono>

namespace ffmpeg {

/**
 * Class keeps the track of the decoded timestamps (us) for media streams.
 */

class TimeKeeper {
 public:
  TimeKeeper() = default;

  // adjust provided @timestamp to the corrected value
  // return advised sleep time before next frame processing in (us)
  long adjust(long& decoderTimestamp);

 private:
  long startTime_{0};
  long streamTimestamp_{0};
};

} // namespace ffmpeg
