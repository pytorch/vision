#pragma once
#include <memory>
#include "FfmpegDecoder.h"

namespace util {

std::unique_ptr<DecoderParameters> getDecoderParams(
    double seekFrameMargin,
    int64_t getPtsOnly,
    size_t videoWidth,
    size_t videoHeight,
    size_t videoMinDimension,
    int64_t videoStartPts,
    int64_t videoEndPts,
    int videoTimeBaseNum,
    int videoTimeBaseDen,
    size_t audioSamples,
    size_t audioChannels,
    int64_t audioStartPts,
    int64_t audioEndPts,
    int audioTimeBaseNum,
    int audioTimeBaseDen);

} // namespace util
