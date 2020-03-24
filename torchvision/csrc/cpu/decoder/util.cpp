#include "util.h"
#include <c10/util/Logging.h>

namespace ffmpeg {

namespace Serializer {

// fixed size types
template <typename T>
inline size_t getSize(const T& x) {
  return sizeof(x);
}

template <typename T>
inline bool serializeItem(
    uint8_t* dest,
    size_t len,
    size_t& pos,
    const T& src) {
  VLOG(6) << "Generic serializeItem";
  const auto required = sizeof(src);
  if (len < pos + required) {
    return false;
  }
  memcpy(dest + pos, &src, required);
  pos += required;
  return true;
}

template <typename T>
inline bool deserializeItem(
    const uint8_t* src,
    size_t len,
    size_t& pos,
    T& dest) {
  const auto required = sizeof(dest);
  if (len < pos + required) {
    return false;
  }
  memcpy(&dest, src + pos, required);
  pos += required;
  return true;
}

// AVSubtitleRect specialization
inline size_t getSize(const AVSubtitleRect& x) {
  auto rectBytes = [](const AVSubtitleRect& y) -> size_t {
    size_t s = 0;
    switch (y.type) {
      case SUBTITLE_BITMAP:
        for (int i = 0; i < y.nb_colors; ++i) {
          s += sizeof(y.pict.linesize[i]);
          s += y.pict.linesize[i];
        }
        break;
      case SUBTITLE_TEXT:
        s += sizeof(size_t);
        s += strlen(y.text);
        break;
      case SUBTITLE_ASS:
        s += sizeof(size_t);
        s += strlen(y.ass);
        break;
      default:
        break;
    }
    return s;
  };
  return getSize(x.x) + getSize(x.y) + getSize(x.w) + getSize(x.h) +
      getSize(x.nb_colors) + getSize(x.type) + getSize(x.flags) + rectBytes(x);
}

// AVSubtitle specialization
inline size_t getSize(const AVSubtitle& x) {
  auto rectBytes = [](const AVSubtitle& y) -> size_t {
    size_t s = getSize(y.num_rects);
    for (unsigned i = 0; i < y.num_rects; ++i) {
      s += getSize(*y.rects[i]);
    }
    return s;
  };
  return getSize(x.format) + getSize(x.start_display_time) +
      getSize(x.end_display_time) + getSize(x.pts) + rectBytes(x);
}

inline bool serializeItem(
    uint8_t* dest,
    size_t len,
    size_t& pos,
    const AVSubtitleRect& src) {
  auto rectSerialize =
      [](uint8_t* d, size_t l, size_t& p, const AVSubtitleRect& x) -> size_t {
    switch (x.type) {
      case SUBTITLE_BITMAP:
        for (int i = 0; i < x.nb_colors; ++i) {
          if (!serializeItem(d, l, p, x.pict.linesize[i])) {
            return false;
          }
          if (p + x.pict.linesize[i] > l) {
            return false;
          }
          memcpy(d + p, x.pict.data[i], x.pict.linesize[i]);
          p += x.pict.linesize[i];
        }
        return true;
      case SUBTITLE_TEXT: {
        const size_t s = strlen(x.text);
        if (!serializeItem(d, l, p, s)) {
          return false;
        }
        if (p + s > l) {
          return false;
        }
        memcpy(d + p, x.text, s);
        p += s;
        return true;
      }
      case SUBTITLE_ASS: {
        const size_t s = strlen(x.ass);
        if (!serializeItem(d, l, p, s)) {
          return false;
        }
        if (p + s > l) {
          return false;
        }
        memcpy(d + p, x.ass, s);
        p += s;
        return true;
      }
      default:
        return true;
    }
  };
  return serializeItem(dest, len, pos, src.x) &&
      serializeItem(dest, len, pos, src.y) &&
      serializeItem(dest, len, pos, src.w) &&
      serializeItem(dest, len, pos, src.h) &&
      serializeItem(dest, len, pos, src.nb_colors) &&
      serializeItem(dest, len, pos, src.type) &&
      serializeItem(dest, len, pos, src.flags) &&
      rectSerialize(dest, len, pos, src);
}

inline bool serializeItem(
    uint8_t* dest,
    size_t len,
    size_t& pos,
    const AVSubtitle& src) {
  auto rectSerialize =
      [](uint8_t* d, size_t l, size_t& p, const AVSubtitle& x) -> bool {
    bool res = serializeItem(d, l, p, x.num_rects);
    for (unsigned i = 0; res && i < x.num_rects; ++i) {
      res = serializeItem(d, l, p, *(x.rects[i]));
    }
    return res;
  };
  VLOG(6) << "AVSubtitle serializeItem";
  return serializeItem(dest, len, pos, src.format) &&
      serializeItem(dest, len, pos, src.start_display_time) &&
      serializeItem(dest, len, pos, src.end_display_time) &&
      serializeItem(dest, len, pos, src.pts) &&
      rectSerialize(dest, len, pos, src);
}

inline bool deserializeItem(
    const uint8_t* src,
    size_t len,
    size_t& pos,
    AVSubtitleRect& dest) {
  auto rectDeserialize =
      [](const uint8_t* y, size_t l, size_t& p, AVSubtitleRect& x) -> bool {
    switch (x.type) {
      case SUBTITLE_BITMAP:
        for (int i = 0; i < x.nb_colors; ++i) {
          if (!deserializeItem(y, l, p, x.pict.linesize[i])) {
            return false;
          }
          if (p + x.pict.linesize[i] > l) {
            return false;
          }
          x.pict.data[i] = (uint8_t*)av_malloc(x.pict.linesize[i]);
          memcpy(x.pict.data[i], y + p, x.pict.linesize[i]);
          p += x.pict.linesize[i];
        }
        return true;
      case SUBTITLE_TEXT: {
        size_t s = 0;
        if (!deserializeItem(y, l, p, s)) {
          return false;
        }
        if (p + s > l) {
          return false;
        }
        x.text = (char*)av_malloc(s + 1);
        memcpy(x.text, y + p, s);
        x.text[s] = 0;
        p += s;
        return true;
      }
      case SUBTITLE_ASS: {
        size_t s = 0;
        if (!deserializeItem(y, l, p, s)) {
          return false;
        }
        if (p + s > l) {
          return false;
        }
        x.ass = (char*)av_malloc(s + 1);
        memcpy(x.ass, y + p, s);
        x.ass[s] = 0;
        p += s;
        return true;
      }
      default:
        return true;
    }
  };

  return deserializeItem(src, len, pos, dest.x) &&
      deserializeItem(src, len, pos, dest.y) &&
      deserializeItem(src, len, pos, dest.w) &&
      deserializeItem(src, len, pos, dest.h) &&
      deserializeItem(src, len, pos, dest.nb_colors) &&
      deserializeItem(src, len, pos, dest.type) &&
      deserializeItem(src, len, pos, dest.flags) &&
      rectDeserialize(src, len, pos, dest);
}

inline bool deserializeItem(
    const uint8_t* src,
    size_t len,
    size_t& pos,
    AVSubtitle& dest) {
  auto rectDeserialize =
      [](const uint8_t* y, size_t l, size_t& p, AVSubtitle& x) -> bool {
    bool res = deserializeItem(y, l, p, x.num_rects);
    if (res && x.num_rects) {
      x.rects =
          (AVSubtitleRect**)av_malloc(x.num_rects * sizeof(AVSubtitleRect*));
    }
    for (unsigned i = 0; res && i < x.num_rects; ++i) {
      x.rects[i] = (AVSubtitleRect*)av_malloc(sizeof(AVSubtitleRect));
      memset(x.rects[i], 0, sizeof(AVSubtitleRect));
      res = deserializeItem(y, l, p, *x.rects[i]);
    }
    return res;
  };
  return deserializeItem(src, len, pos, dest.format) &&
      deserializeItem(src, len, pos, dest.start_display_time) &&
      deserializeItem(src, len, pos, dest.end_display_time) &&
      deserializeItem(src, len, pos, dest.pts) &&
      rectDeserialize(src, len, pos, dest);
}
} // namespace Serializer

namespace Util {
std::string generateErrorDesc(int errorCode) {
  std::array<char, 1024> buffer;
  if (av_strerror(errorCode, buffer.data(), buffer.size()) < 0) {
    return std::string("Unknown error code: ") + std::to_string(errorCode);
  }
  buffer.back() = 0;
  return std::string(buffer.data());
}

size_t serialize(const AVSubtitle& sub, ByteStorage* out) {
  const auto len = size(sub);
  CHECK_LE(len, out->tail());
  size_t pos = 0;
  if (!Serializer::serializeItem(out->writableTail(), len, pos, sub)) {
    return 0;
  }
  out->append(len);
  return len;
}

bool deserialize(const ByteStorage& buf, AVSubtitle* sub) {
  size_t pos = 0;
  return Serializer::deserializeItem(buf.data(), buf.length(), pos, *sub);
}

size_t size(const AVSubtitle& sub) {
  return Serializer::getSize(sub);
}

bool validateVideoFormat(const VideoFormat& f) {
  // clang-format off
  /*
  Valid parameters values for decoder
  ____________________________________________________________________________________
  |  W  |  H  | minDimension | maxDimension | cropImage |  algorithm                 |
  |__________________________________________________________________________________|
  |  0  |  0  |     0        |  0           |  N/A      |   original                 |
  |__________________________________________________________________________________|
  |  >0 |  0  |     N/A      |  N/A         |  N/A      |   scale keeping W          |
  |__________________________________________________________________________________|
  |  0  |  >0 |     N/A      |  N/A         |  N/A      |   scale keeping H          |
  |__________________________________________________________________________________|
  |  >0 |  >0 |     N/A      |  N/A         |  0        |   stretch/scale            |
  |__________________________________________________________________________________|
  |  >0 |  >0 |     N/A      |  N/A         |  >0       |   scale/crop               |
  |__________________________________________________________________________________|
  |  0  |  0  |     >0       |  0           |  N/A      |scale to min dimension      |
  |__________________________________________________________________________________|
  |  0  |  0  |     0        |  >0          |  N/A      |scale to max dimension      |
  |__________________________________________________________________________________|
  |  0  |  0  |     >0       |  >0          |  N/A      |stretch to min/max dimension|
  |_____|_____|______________|______________|___________|____________________________|

  */
  // clang-format on
  return (f.width == 0 && // #1, #6, #7 and #8
          f.height == 0 && f.cropImage == 0) ||
      (f.width != 0 && // #4 and #5
       f.height != 0 && f.minDimension == 0 && f.maxDimension == 0) ||
      (((f.width != 0 && // #2
         f.height == 0) ||
        (f.width == 0 && // #3
         f.height != 0)) &&
       f.minDimension == 0 && f.maxDimension == 0 && f.cropImage == 0);
}

void setFormatDimensions(
    size_t& destW,
    size_t& destH,
    size_t userW,
    size_t userH,
    size_t srcW,
    size_t srcH,
    size_t minDimension,
    size_t maxDimension,
    size_t cropImage) {
  // rounding rules
  // int -> double -> round up
  // if fraction is >= 0.5 or round down if fraction is < 0.5
  // int result = double(value) + 0.5
  // here we rounding double to int according to the above rule

  // #1, #6, #7 and #8
  if (userW == 0 && userH == 0) {
    if (minDimension > 0 && maxDimension == 0) { // #6
      if (srcW > srcH) {
        // landscape
        destH = minDimension;
        destW = round(double(srcW * minDimension) / srcH);
      } else {
        // portrait
        destW = minDimension;
        destH = round(double(srcH * minDimension) / srcW);
      }
    } else if (minDimension == 0 && maxDimension > 0) { // #7
      if (srcW > srcH) {
        // landscape
        destW = maxDimension;
        destH = round(double(srcH * maxDimension) / srcW);
      } else {
        // portrait
        destH = maxDimension;
        destW = round(double(srcW * maxDimension) / srcH);
      }
    } else if (minDimension > 0 && maxDimension > 0) { // #8
      if (srcW > srcH) {
        // landscape
        destW = maxDimension;
        destH = minDimension;
      } else {
        // portrait
        destW = minDimension;
        destH = maxDimension;
      }
    } else { // #1
      destW = srcW;
      destH = srcH;
    }
  } else if (userW != 0 && userH == 0) { // #2
    destW = userW;
    destH = round(double(srcH * userW) / srcW);
  } else if (userW == 0 && userH != 0) { // #3
    destW = round(double(srcW * userH) / srcH);
    destH = userH;
  } else { // userW != 0 && userH != 0
    if (cropImage == 0) { // #4
      destW = userW;
      destH = userH;
    } else { // #5
      double userSlope = double(userH) / userW;
      double srcSlope = double(srcH) / srcW;
      if (srcSlope < userSlope) {
        destW = round(double(srcW * userH) / srcH);
        destH = userH;
      } else {
        destW = userW;
        destH = round(double(srcH * userW) / srcW);
      }
    }
  }
  // prevent zeros
  destW = std::max(destW, 1UL);
  destH = std::max(destH, 1UL);
}
} // namespace Util
} // namespace ffmpeg
