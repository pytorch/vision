#pragma once

#include <string>
#include <array>
#include "FfmpegHeaders.h"
#include "Interface.h"

namespace ffmpeg_util {

bool mapFfmpegType(AVMediaType media, enum MediaType* type);

bool mapMediaType(MediaType type, enum AVMediaType* media);

void setFormatDimensions(
    size_t& destW,
    size_t& destH,
    size_t userW,
    size_t userH,
    size_t srcW,
    size_t srcH,
    size_t minDimension);

bool validateVideoFormat(const VideoFormat& f);

std::string getErrorDesc(int errnum);

} // namespace ffmpeg_util
