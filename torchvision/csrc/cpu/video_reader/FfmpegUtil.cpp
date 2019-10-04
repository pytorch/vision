#include "FfmpegUtil.h"

using namespace std;

namespace ffmpeg_util {

bool mapFfmpegType(AVMediaType media, MediaType* type) {
  switch (media) {
    case AVMEDIA_TYPE_VIDEO:
      *type = MediaType::TYPE_VIDEO;
      return true;
    case AVMEDIA_TYPE_AUDIO:
      *type = MediaType::TYPE_AUDIO;
      return true;
    default:
      return false;
  }
}

bool mapMediaType(MediaType type, AVMediaType* media) {
  switch (type) {
    case MediaType::TYPE_VIDEO:
      *media = AVMEDIA_TYPE_VIDEO;
      return true;
    case MediaType::TYPE_AUDIO:
      *media = AVMEDIA_TYPE_AUDIO;
      return true;
    default:
      return false;
  }
}

void setFormatDimensions(
    int& destW,
    int& destH,
    int userW,
    int userH,
    int srcW,
    int srcH,
    int minDimension) {
  // rounding rules
  // int -> double -> round
  // round up if fraction is >= 0.5 or round down if fraction is < 0.5
  // int result = double(value) + 0.5
  // here we rounding double to int according to the above rule
  if (userW == 0 && userH == 0) {
    if (minDimension > 0) { // #2
      if (srcW > srcH) {
        // landscape
        destH = minDimension;
        destW = round(double(srcW * minDimension) / srcH);
      } else {
        // portrait
        destW = minDimension;
        destH = round(double(srcH * minDimension) / srcW);
      }
    } else { // #1
      destW = srcW;
      destH = srcH;
    }
  } else if (userW != 0 && userH == 0) { // #3
    destW = userW;
    destH = round(double(srcH * userW) / srcW);
  } else if (userW == 0 && userH != 0) { // #4
    destW = round(double(srcW * userH) / srcH);
    destH = userH;
  } else {
    // userW != 0 && userH != 0. #5
    destW = userW;
    destH = userH;
  }
  // prevent zeros
  destW = std::max(destW, 1);
  destH = std::max(destH, 1);
}

bool validateVideoFormat(const VideoFormat& f) {
  /*
  Valid parameters values for decoder
  ___________________________________________________
  |  W  |  H  | minDimension |  algorithm           |
  |_________________________________________________|
  |  0  |  0  |     0        |   original           |
  |_________________________________________________|
  |  0  |  0  |     >0       |scale to min dimension|
  |_____|_____|____________________________________ |
  |  >0 |  0  |     0        |   scale keeping W    |
  |_________________________________________________|
  |  0  |  >0 |     0        |   scale keeping H    |
  |_________________________________________________|
  |  >0 |  >0 |     0        |   stretch/scale      |
  |_________________________________________________|

  */
  return (f.width == 0 && f.height == 0) || // #1 and #2
      (f.width != 0 && f.height != 0 && f.minDimension == 0) || // # 5
      (((f.width != 0 && f.height == 0) || // #3 and #4
        (f.width == 0 && f.height != 0)) &&
       f.minDimension == 0);
}

string getErrorDesc(int errnum) {
  array<char, 1024> buffer;
  if (av_strerror(errnum, buffer.data(), buffer.size()) < 0) {
    return string("Unknown error code");
  }
  buffer.back() = 0;
  return string(buffer.data());
}

} // namespace ffmpeg_util
