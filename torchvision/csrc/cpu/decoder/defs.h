#pragma once

#include <array>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
#include "libswscale/swscale.h"
}

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
  long format{-1}; // AVSampleFormat, auto AV_SAMPLE_FMT_NONE
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
  /*
  When width = 0, height = 0, minDimension = 0, and maxDimension = 0,
    keep the orignal frame resolution
  When width = 0, height = 0, minDimension != 0, and maxDimension = 0,
    keep the aspect ratio and resize the frame so that shorter edge size is
  minDimension
  When width = 0, height = 0, minDimension = 0, and maxDimension != 0,
    keep the aspect ratio and resize the frame so that longer edge size is
  maxDimension
  When width = 0, height = 0, minDimension != 0, and maxDimension != 0,
    resize the frame so that shorter edge size is minDimension, and
    longer edge size is maxDimension. The aspect ratio may not be preserved
  When width = 0, height != 0, minDimension = 0, and maxDimension = 0,
    keep the aspect ratio and resize the frame so that frame height is $height
  When width != 0, height = 0, minDimension = 0, and maxDimension = 0,
    keep the aspect ratio and resize the frame so that frame width is $width
  When width != 0, height != 0, minDimension = 0, and maxDimension = 0,
    resize the frame so that frame width and  height are set to $width and
  $height,
    respectively
  */
  size_t width{0}; // width in pixels
  size_t height{0}; // height in pixels
  long format{-1}; // AVPixelFormat, auto AV_PIX_FMT_NONE
  size_t minDimension{0}; // choose min dimension and rescale accordingly
  size_t maxDimension{0}; // choose max dimension and rescale accordingly
  size_t cropImage{0}; // request image crop
  // -- alignment 40 bytes
};

// subtitle/cc
struct SubtitleFormat {
  long type{0}; // AVSubtitleType, auto SUBTITLE_NONE
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

  explicit MediaFormat(long s = -1) : type(TYPE_AUDIO), stream(s), format() {}
  explicit MediaFormat(int x, long s = -1)
      : type(TYPE_VIDEO), stream(s), format(x) {}
  explicit MediaFormat(char x, long s = -1)
      : type(TYPE_SUBTITLE), stream(s), format(x) {}
  explicit MediaFormat(double x, long s = -1)
      : type(TYPE_CC), stream(s), format(x) {}

  static MediaFormat makeMediaFormat(AudioFormat format, long stream) {
    MediaFormat result(stream);
    result.format.audio = format;
    return result;
  }

  static MediaFormat makeMediaFormat(VideoFormat format, long stream) {
    MediaFormat result(0, stream);
    result.format.video = format;
    return result;
  }

  static MediaFormat makeMediaFormat(SubtitleFormat format, long stream) {
    MediaFormat result('0', stream);
    result.format.subtitle = format;
    return result;
  }

  // format type
  MediaType type;
  // stream index:
  // set -1 for one stream auto detection, -2 for all streams auto detection,
  // >= 0, specified stream, if caller knows the stream index (unlikely)
  long stream;
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
  long logLevel{0};
  // when decoder would give up, 0 means never
  size_t maxPackageErrors{0};
  // max allowed consecutive times no bytes are processed. 0 means for infinite.
  size_t maxProcessNoBytes{0};
  // start offset (us)
  long startOffset{0};
  // end offset (us)
  long endOffset{-1};
  // logging id
  int64_t loggingUuid{0};
  // internal max seekable buffer size
  size_t maxSeekableBytes{0};
  // adjust header pts to the epoch time
  bool convertPtsToWallTime{false};
  // indicate if input stream is an encoded image
  bool isImage{false};
  // listen and wait for new rtmp stream
  bool listen{false};
  // don't copy frame body, only header
  bool headerOnly{false};
  // interrupt init method on timeout
  bool preventStaleness{true};
  // seek tolerated accuracy (us)
  double seekAccuracy{1000000.0};
  // what media types should be processed, default none
  std::set<MediaFormat> formats;

  // can be used for asynchronous decoders
  size_t cacheSize{8192}; // mow many bytes to cache before stop reading bytes
  size_t cacheTimeoutMs{1000}; // timeout on bytes writing
  bool enforceCacheSize{false}; // drop output frames if cache is full
  bool mergeAudioMessages{false}; // combine collocated audio messages together
};

struct DecoderHeader {
  // message id, from 0 till ...
  size_t seqno{0};
  // decoded timestamp in microseconds from either beginning of the stream or
  // from epoch time, see DecoderParameters::convertPtsToWallTime
  long pts{0};
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

/*
 * External provider of the ecnoded bytes, specific implementation is left for
 * different use cases, like file, memory, external network end-points, etc.
 * Normally input/output parameter @out set to valid, not null buffer pointer,
 * which indicates "read" call, however there are "seek" modes as well.

 * @out != nullptr => read from the current offset, @whence got ignored,
 * @size bytes to read => return number bytes got read, 0 if no more bytes
 * available, < 0 on error.

 * @out == nullptr, @timeoutMs == 0 => does provider support "seek"
 * capability in a first place? @size & @whence got ignored, return 0 on
 * success, < 0 if "seek" mode is not supported.

 * @out == nullptr, @timeoutMs != 0 => normal seek call
 * offset == @size, i.e. @whence = [SEEK_SET, SEEK_CUR, SEEK_END, AVSEEK_SIZE)
 * return < 0 on error, position if @whence = [SEEK_SET, SEEK_CUR, SEEK_END],
 * length of buffer if @whence = [AVSEEK_SIZE].
 */
using DecoderInCallback =
    std::function<int(uint8_t* out, int size, int whence, uint64_t timeoutMs)>;

using DecoderOutCallback = std::function<void(DecoderOutputMessage&&)>;

struct DecoderMetadata {
  // time base numerator
  long num{0};
  // time base denominator
  long den{1};
  // duration of the stream, in miscroseconds, if available
  long duration{-1};
  // frames per second, valid only for video streams
  double fps{0};
  // format specifies what kind frame is in a payload
  MediaFormat format;
};
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
   * Caller can provide non-null pointer for the input container
   * if headers to obtain the streams metadata (optional)
   */
  virtual bool init(
      const DecoderParameters& params,
      DecoderInCallback&& in,
      std::vector<DecoderMetadata>* metadata) = 0;

  /**
   * Polls available decoded one frame from decoder
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
   * Interrupts whatever decoder is doing at any time
   */
  virtual void interrupt() = 0;

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

  /**
   * Releases resources
   */
  virtual void shutdown() = 0;

  /*
   * Returns media type
   */
  MediaType getMediaType() const {
    return params_.type;
  }
  /*
   * Returns formats
   */
  FormatUnion getInputFormat() const {
    return params_.in;
  }
  FormatUnion getOutFormat() const {
    return params_.out;
  }

 protected:
  SamplerParameters params_;
};
} // namespace ffmpeg
