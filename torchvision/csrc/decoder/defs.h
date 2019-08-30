#pragma once

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace ffmpeg {

// bit mask of formats, keep them in form 2^n
enum MediaType : size_t {
  TYPE_AUDIO = 1,
  TYPE_VIDEO = 2,
  TYPE_SUBTITLE = 4,
  TYPE_CC = 8, // closed captions from transport streams
};

// audio
struct AudioFormat {
  // fields are initialized for the auto detection
  // caller can specify some/all of field values if specific output is desirable
  bool operator==(const AudioFormat& x) const {
    return x.format == format && x.samples == samples && x.channels == channels;
  }

  size_t samples{0}; // number samples per second (frequency)
  size_t channels{0}; // number of channels
  ssize_t format{-1}; // AVSampleFormat, auto AV_SAMPLE_FMT_NONE
  size_t padding[2];
  // -- alignment 40 bytes
};

// video
struct VideoFormat {
  // fields are initialized for the auto detection
  // caller can specify some/all of field values if specific output is desirable
  bool operator==(const VideoFormat& x) const {
    return x.format == format && x.width == width && x.height == height;
  }

  size_t width{0}; // width in pixels
  size_t height{0}; // height in pixels
  ssize_t format{-1}; // AVPixelFormat, auto AV_PIX_FMT_NONE
  size_t minDimension{0}; // choose min dimension and rescale accordingly
  size_t cropImage{0}; // request image crop
  // -- alignment 40 bytes
};

// subtitle/cc
struct SubtitleFormat {
  ssize_t type{0}; // AVSubtitleType, auto SUBTITLE_NONE
  size_t padding[4];
  // -- alignment 40 bytes
};

union FormatUnion {
  FormatUnion() : audio() {}
  explicit FormatUnion(int) : video() {}
  explicit FormatUnion(char) : subtitle() {}
  explicit FormatUnion(double) : subtitle() {}
  AudioFormat audio;
  VideoFormat video;
  SubtitleFormat subtitle;
  // -- alignment 40 bytes
};

/*
  MediaFormat data structure serves as input/output parameter.
  Caller assigns values for input formats
  or leave default values for auto detection
  For output formats all fields will be set to the specific values
*/

struct MediaFormat {
  // for using map/set data structures
  bool operator<(const MediaFormat& x) const {
    return type < x.type;
  }
  bool operator==(const MediaFormat& x) const {
    if (type != x.type) {
      return false;
    }
    switch (type) {
      case TYPE_AUDIO:
        return format.audio == x.format.audio;
      case TYPE_VIDEO:
        return format.video == x.format.video;
      case TYPE_SUBTITLE:
      case TYPE_CC:
        return true;
      default:
        return false;
    }
  }

  MediaFormat() : type(TYPE_AUDIO), stream(-1), format() {}
  explicit MediaFormat(int x) : type(TYPE_VIDEO), stream(-1), format(x) {}
  explicit MediaFormat(char x) : type(TYPE_SUBTITLE), stream(-1), format(x) {}
  explicit MediaFormat(double x) : type(TYPE_CC), stream(-1), format(x) {}
  // format type
  MediaType type;
  // stream index:
  // set -1 for one stream auto detection, -2 for all streams auto detection,
  // >= 0, specified stream, if caller knows the stream index (unlikely)
  ssize_t stream;
  // union keeps one of the possible formats, defined by MediaType
  FormatUnion format;
};

struct DecoderParameters {
  // local file, remote file, http url, rtmp stream uri, etc. anything that
  // ffmpeg can recognize
  std::string uri;
  // timeout on getting bytes for decoding
  size_t timeoutMs{1000};
  // logging level, default AV_LOG_PANIC
  ssize_t logLevel{0};
  // when decoder would give up, 0 means never
  size_t maxPackageErrors{0};
  // max allowed consecutive times no bytes are processed. 0 means for infinite.
  size_t maxProcessNoBytes{0};
  // logging id
  int64_t loggingUuid{0};
  // adjust header pts to the epoch time
  bool convertPtsToWallTime{false};
  // indicate if input stream is an encoded image
  bool isImage{false};
  // what media types should be processed, default none
  std::set<MediaFormat> formats;
};

struct DecoderHeader {
  // message id, from 0 till ...
  size_t seqno{0};
  // decoded timestamp in microseconds from either beginning of the stream or
  // from epoch time, see DecoderParameters::convertPtsToWallTime
  ssize_t pts{0};
  // decoded key frame
  size_t keyFrame{0};
  // frames per second, valid only for video streams
  double fps{0};
  // format specifies what kind frame is in a payload
  MediaFormat format;
};

// Abstract interface ByteStorage class
class ByteStorage {
 public:
  virtual ~ByteStorage() = default;
  // makes sure that buffer has at least n bytes available for writing, if not
  // storage must reallocate memory.
  virtual void ensure(size_t n) = 0;
  // caller must not to write more than available bytes
  virtual uint8_t* writableTail() = 0;
  // caller confirms that n bytes were written to the writable tail
  virtual void append(size_t n) = 0;
  // caller confirms that n bytes were read from the read buffer
  virtual void trim(size_t n) = 0;
  // gives an access to the beginning of the read buffer
  virtual const uint8_t* data() const = 0;
  // returns the stored size in bytes
  virtual size_t length() const = 0;
  // returns available capacity for writable tail
  virtual size_t tail() const = 0;
  // clears content, keeps capacity
  virtual void clear() = 0;
};

struct DecoderOutputMessage {
  DecoderHeader header;
  std::unique_ptr<ByteStorage> payload;
};

using DecoderInCallback =
    std::function<int(uint8_t* out, int size, uint64_t timeoutMs)>;

using DecoderOutCallback = std::function<void(DecoderOutputMessage&&)>;

/**
 * Abstract class for decoding media bytes
 * It has two diffrent modes. Internal media bytes retrieval for given uri and
 * external media bytes provider in case of memory streams
 */
class MediaDecoder {
 public:
  virtual ~MediaDecoder() = default;

  /**
   * Initializes media decoder with parameters,
   * calls callback when media bytes are available.
   * Media bytes get fetched internally from provided URI
   * or invokes provided input callback to get media bytes.
   * Input callback must be empty for the internal media provider
   */
  virtual bool init(
      const DecoderParameters& params,
      DecoderInCallback&& in) = 0;

  /**
   * Polls available decoded bytes from decoder
   * Returns error code, 0 - for success
   */
  virtual int decode(DecoderOutputMessage* out, uint64_t timeoutMs) = 0;

  /**
   * Polls available decoded bytes from decoder, till EOF or error
   */
  virtual int decode_all(const DecoderOutCallback& callback) = 0;

  /**
   * Stops calling callback, releases resources
   */
  virtual void shutdown() = 0;

  /**
   * Factory to create ByteStorage class instances, particular implementation is
   * left to the derived class. Caller provides the initially allocated size
   */
  virtual std::unique_ptr<ByteStorage> createByteStorage(size_t n) = 0;
};

struct SamplerParameters {
  MediaType type{TYPE_AUDIO};
  FormatUnion in;
  FormatUnion out;
  int64_t loggingUuid{0};
};

/**
 * Abstract class for sampling media bytes
 */
class MediaSampler {
 public:
  virtual ~MediaSampler() = default;

  /**
   * Initializes media sampler with parameters
   */
  virtual bool init(const SamplerParameters& params) = 0;

  /**
   * Samples media bytes
   * Returns error code < 0, or >=0 - for success, indicating number of bytes
   * processed.
   * set @in to null for flushing data
   */
  virtual int sample(const ByteStorage* in, ByteStorage* out) = 0;

  /*
    Returns media type
   */
  virtual MediaType getMediaType() const = 0;

  /*
   Returns formats
   */
  virtual FormatUnion getInputFormat() const = 0;
  virtual FormatUnion getOutFormat() const = 0;

  /**
   * Releases resources
   */
  virtual void shutdown() = 0;
};
} // namespace ffmpeg
