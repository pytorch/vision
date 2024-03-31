#include "time_keeper.h"
#include "defs.h"

namespace ffmpeg {

namespace {
const long kMaxTimeBaseDiference = 10;
}

long TimeKeeper::adjust(long& decoderTimestamp) {
  const long now = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();

  if (startTime_ == 0) {
    startTime_ = now;
  }
  if (streamTimestamp_ == 0) {
    streamTimestamp_ = decoderTimestamp;
  }

  const auto runOut = startTime_ + decoderTimestamp - streamTimestamp_;

  if (std::labs((now - runOut) / AV_TIME_BASE) > kMaxTimeBaseDiference) {
    streamTimestamp_ = startTime_ - now + decoderTimestamp;
  }

  const auto sleepAdvised = runOut - now;

  decoderTimestamp += startTime_ - streamTimestamp_;

  return sleepAdvised > 0 ? sleepAdvised : 0;
}

} // namespace ffmpeg
