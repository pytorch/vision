#pragma once

#include <c10/util/Logging.h>
#include <sys/types.h>
#include <memory>
#include <unordered_map>

extern "C" {

#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
void av_free(void* ptr);
}

struct avDeleter {
  void operator()(uint8_t* p) const {
    av_free(p);
  }
};

const AVPixelFormat defaultVideoPixelFormat = AV_PIX_FMT_RGB24;
const AVSampleFormat defaultAudioSampleFormat = AV_SAMPLE_FMT_FLT;

using AvDataPtr = std::unique_ptr<uint8_t, avDeleter>;

enum MediaType : uint32_t {
  TYPE_VIDEO = 1,
  TYPE_AUDIO = 2,
};

struct EnumClassHash {
  template <typename T>
  uint32_t operator()(T t) const {
    return static_cast<uint32_t>(t);
  }
};

struct VideoFormat {
  // fields are initialized for the auto detection
  // caller can specify some/all of field values if specific output is desirable

  int width{0}; // width in pixels
  int height{0}; // height in pixels
  int minDimension{0}; // choose min dimension and rescale accordingly
  // Output image pixel format. data type AVPixelFormat
  AVPixelFormat format{defaultVideoPixelFormat}; // type AVPixelFormat
  int64_t startPts{0}, endPts{0}; // Start and end presentation timestamp
  int timeBaseNum{0};
  int timeBaseDen{1}; // numerator and denominator of time base
  float fps{0.0};
  int64_t duration{0}; // duration of the stream, in stream time base
};

struct AudioFormat {
  // fields are initialized for the auto detection
  // caller can specify some/all of field values if specific output is desirable

  int samples{0}; // number samples per second (frequency)
  int channels{0}; // number of channels
  AVSampleFormat format{defaultAudioSampleFormat}; // type AVSampleFormat
  int64_t startPts{0}, endPts{0}; // Start and end presentation timestamp
  int timeBaseNum{0};
  int timeBaseDen{1}; // numerator and denominator of time base
  int64_t duration{0}; // duration of the stream, in stream time base
};

union FormatUnion {
  FormatUnion() {}
  VideoFormat video;
  AudioFormat audio;
};

struct MediaFormat {
  MediaFormat() {}

  MediaFormat(const MediaFormat& mediaFormat) : type(mediaFormat.type) {
    if (type == MediaType::TYPE_VIDEO) {
      format.video = mediaFormat.format.video;
    } else if (type == MediaType::TYPE_AUDIO) {
      format.audio = mediaFormat.format.audio;
    }
  }

  MediaFormat(MediaType mediaType) : type(mediaType) {
    if (mediaType == MediaType::TYPE_VIDEO) {
      format.video = VideoFormat();
    } else if (mediaType == MediaType::TYPE_AUDIO) {
      format.audio = AudioFormat();
    }
  }
  // media type
  MediaType type;
  // format data
  FormatUnion format;
};

class DecodedFrame {
 public:
  explicit DecodedFrame() : frame_(nullptr), frameSize_(0), pts_(0) {}
  explicit DecodedFrame(AvDataPtr frame, int frameSize, int64_t pts)
      : frame_(std::move(frame)), frameSize_(frameSize), pts_(pts) {}
  AvDataPtr frame_{nullptr};
  int frameSize_{0};
  int64_t pts_{0};
};

struct MediaData {
  MediaData() {}
  MediaData(FormatUnion format) : format_(format) {}
  FormatUnion format_;
  std::vector<std::unique_ptr<DecodedFrame>> frames_;
};

class DecoderOutput {
 public:
  explicit DecoderOutput() {}

  ~DecoderOutput() {}

  void initMediaType(MediaType mediaType, FormatUnion format);

  void addMediaFrame(MediaType mediaType, std::unique_ptr<DecodedFrame> frame);

  void clear();

  std::unordered_map<MediaType, MediaData, EnumClassHash> media_data_;
};
