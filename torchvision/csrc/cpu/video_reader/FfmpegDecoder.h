#pragma once

#include <string>
#include <vector>

#include "FfmpegHeaders.h"
#include "FfmpegStream.h"
#include "Interface.h"

#define VIO_BUFFER_SZ 81920
#define AVPROBE_SIZE 8192

class DecoderParameters {
 public:
  std::unordered_map<MediaType, MediaFormat, EnumClassHash> formats;
  // av_seek_frame is imprecise so seek to a timestamp earlier by a margin
  // The unit of margin is second
  double seekFrameMargin{1.0};
  // When getPtsOnly is set to 1, we only get pts of each frame and don not
  // output frame data. It will be much faster
  int64_t getPtsOnly{0};
};

class FfmpegAvioContext {
 public:
  FfmpegAvioContext();

  int initAVIOContext(const uint8_t* buffer, int64_t size);

  ~FfmpegAvioContext();

  int read(uint8_t* buf, int buf_size);

  static int readMemory(void* opaque, uint8_t* buf, int buf_size);

  int64_t seek(int64_t offset, int whence);

  static int64_t seekMemory(void* opaque, int64_t offset, int whence);

  AVIOContext* get_avio() {
    return avioCtx_;
  }

 private:
  int workBuffersize_;
  uint8_t* workBuffer_;
  // for file mode
  FILE* inputFile_;
  // for memory mode
  const uint8_t* inputBuffer_;
  int inputBufferSize_;
  int offset_ = 0;

  AVIOContext* avioCtx_{nullptr};
};

class FfmpegDecoder {
 public:
  FfmpegDecoder() {
    av_register_all();
  }
  ~FfmpegDecoder() {
    cleanUp();
  }
  // return 0 on success
  // return negative number on failure
  int decodeFile(
      std::unique_ptr<DecoderParameters> params,
      const std::string& filename,
      DecoderOutput& decoderOutput);
  // return 0 on success
  // return negative number on failure
  int decodeMemory(
      std::unique_ptr<DecoderParameters> params,
      const uint8_t* buffer,
      int64_t size,
      DecoderOutput& decoderOutput);
  // return 0 on success
  // return negative number on failure
  int probeFile(
      std::unique_ptr<DecoderParameters> params,
      const std::string& filename,
      DecoderOutput& decoderOutput);
  // return 0 on success
  // return negative number on failure
  int probeMemory(
      std::unique_ptr<DecoderParameters> params,
      const uint8_t* buffer,
      int64_t size,
      DecoderOutput& decoderOutput);

  void cleanUp();

 private:
  FfmpegStream* findStreamByIndex(int streamIndex) const;

  int init(
      const std::string& filename,
      bool isDecodeFile,
      FfmpegAvioContext& ioctx,
      DecoderOutput& decoderOutput);
  // return 0 on success
  // return negative number on failure
  int decodeLoop(
      std::unique_ptr<DecoderParameters> params,
      const std::string& filename,
      bool isDecodeFile,
      FfmpegAvioContext& ioctx,
      DecoderOutput& decoderOutput);

  int probeVideo(
      std::unique_ptr<DecoderParameters> params,
      const std::string& filename,
      bool isDecodeFile,
      FfmpegAvioContext& ioctx,
      DecoderOutput& decoderOutput);

  bool initStreams();

  void flushStreams(DecoderOutput& decoderOutput);
  // whether in all streams, the pts of most recent frame exceeds range
  bool isPtsExceedRange();

  std::unordered_map<int, std::unique_ptr<FfmpegStream>> streams_;
  AVFormatContext* formatCtx_{nullptr};
  std::unique_ptr<DecoderParameters> params_{nullptr};
};
