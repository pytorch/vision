#pragma once

#include <bitset>
#include <unordered_map>
#include "seekable_buffer.h"
#include "stream.h"

#if defined(_MSC_VER)
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif

namespace ffmpeg {

/**
 * Class uses FFMPEG library to decode media streams.
 * Media bytes can be explicitly provided through read-callback
 * or fetched internally by FFMPEG library
 */
class Decoder : public MediaDecoder {
 public:
  Decoder();
  ~Decoder() override;

  // MediaDecoder overrides
  bool init(
      const DecoderParameters& params,
      DecoderInCallback&& in,
      std::vector<DecoderMetadata>* metadata) override;
  int decode_all(const DecoderOutCallback& callback) override;
  void shutdown() override;
  void interrupt() override;

 protected:
  // function does actual work, derived class calls it in working thread
  // periodically. On success method returns 0, ENOADATA on EOF, ETIMEDOUT if
  // no frames got decoded in the specified timeout time, and error on
  // unrecoverable error.
  int getFrame(size_t workingTimeInMs = 100);

  // Derived class must override method and consume the provided message
  virtual void push(DecoderOutputMessage&& buffer) = 0;

  // Fires on init call
  virtual void onInit() {}

 public:
  // C-style FFMPEG API requires C/static methods for callbacks
  static void logFunction(void* avcl, int level, const char* cfmt, va_list vl);
  static int shutdownFunction(void* ctx);
  static int readFunction(void* opaque, uint8_t* buf, int size);
  static int64_t seekFunction(void* opaque, int64_t offset, int whence);
  // can be called by any classes or API
  static void initOnce();

  int* getPrintPrefix() {
    return &printPrefix;
  }

 private:
  // mark below function for a proper invocation
  virtual bool enableLogLevel(int level) const;
  virtual void logCallback(int level, const std::string& message);
  virtual int readCallback(uint8_t* buf, int size);
  virtual int64_t seekCallback(int64_t offset, int whence);
  virtual int shutdownCallback();

  bool openStreams(std::vector<DecoderMetadata>* metadata);
  Stream* findByIndex(int streamIndex) const;
  Stream* findByType(const MediaFormat& format) const;
  int processPacket(
      Stream* stream,
      AVPacket* packet,
      bool* gotFrame,
      bool* hasMsg);
  void flushStreams();
  void cleanUp();

 protected:
  DecoderParameters params_;

 private:
  SeekableBuffer seekableBuffer_;
  int printPrefix{1};

  std::atomic<bool> interrupted_{false};
  AVFormatContext* inputCtx_{nullptr};
  AVIOContext* avioCtx_{nullptr};
  std::unordered_map<ssize_t, std::unique_ptr<Stream>> streams_;
  std::bitset<64> inRange_;
};
} // namespace ffmpeg
