#pragma once

#include <array>
#include <string>
#include "FfmpegHeaders.h"
#include "Interface.h"

namespace ffmpeg_util {

bool mapFfmpegType(AVMediaType media, enum MediaType* type);

bool mapMediaType(MediaType type, enum AVMediaType* media);

void setFormatDimensions(
    int& destW,
    int& destH,
    int userW,
    int userH,
    int srcW,
    int srcH,
    int minDimension);

bool validateVideoFormat(const VideoFormat& f);

std::string getErrorDesc(int errnum);

} // namespace ffmpeg_util
