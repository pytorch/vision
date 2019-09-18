#pragma once
#include <memory>
#include "FfmpegDecoder.h"

namespace util {

std::unique_ptr<DecoderParameters> getDecoderParams(
    double seekFrameMargin,
    int64_t getPtsOnly,
    int64_t readVideoStream,
    int videoWidth,
    int videoHeight,
    int videoMinDimension,
    int64_t videoStartPts,
    int64_t videoEndPts,
    int videoTimeBaseNum,
    int videoTimeBaseDen,
    int64_t readAudioStream,
    int audioSamples,
    int audioChannels,
    int64_t audioStartPts,
    int64_t audioEndPts,
    int audioTimeBaseNum,
    int audioTimeBaseDen);

} // namespace util
