#include <c10/util/Logging.h>
#include <gtest/gtest.h>
#include "memory_buffer.h"
#include "sync_decoder.h"

using namespace ffmpeg;

TEST(SyncDecoder, Test) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.seekAccuracy = 100000;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};
  params.uri = "pytorch/vision/test/assets/videos/R6llTwEh07w.mp4";
  CHECK(decoder.init(params, nullptr));
  DecoderOutputMessage out;
  while (0 == decoder.decode(&out, 100)) {
    LOG(INFO) << "Decoded frame, timestamp(us): " << out.header.pts;
  }
  decoder.shutdown();
}

TEST(SyncDecoder, TestHeadersOnly) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.seekAccuracy = 100000;
  params.headerOnly = true;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};
  params.uri = "pytorch/vision/test/assets/videos/R6llTwEh07w.mp4";
  CHECK(decoder.init(params, nullptr));
  DecoderOutputMessage out;
  while (0 == decoder.decode(&out, 100)) {
    LOG(INFO) << "Decoded frame, type: " << out.header.format.type
              << ", timestamp(us): " << out.header.pts;
  }
  decoder.shutdown();
}

TEST(SyncDecoder, TestMemoryBuffer) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.endOffset = 9000000;
  params.seekAccuracy = 10000;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};

  FILE* f = fopen(
      "pytorch/vision/test/assets/videos/RATRACE_wave_f_nm_np1_fr_goo_37.avi",
      "rb");
  CHECK(f != nullptr);
  fseek(f, 0, SEEK_END);
  std::vector<uint8_t> buffer(ftell(f));
  rewind(f);
  CHECK_EQ(buffer.size(), fread(buffer.data(), 1, buffer.size(), f));
  fclose(f);
  CHECK(decoder.init(
      params, MemoryBuffer::getCallback(buffer.data(), buffer.size())));
  LOG(INFO) << "Decoding from memory bytes: " << buffer.size();
  DecoderOutputMessage out;
  size_t audioFrames = 0, videoFrames = 0;
  while (0 == decoder.decode(&out, 100)) {
    if (out.header.format.type == TYPE_AUDIO) {
      ++audioFrames;
    } else if (out.header.format.type == TYPE_VIDEO) {
      ++videoFrames;
    }
  }
  LOG(INFO) << "Decoded audio frames: " << audioFrames
            << ", video frames: " << videoFrames;
  decoder.shutdown();
}

TEST(SyncDecoder, TestMemoryBufferNoSeekableWithFullRead) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.endOffset = 9000000;
  params.seekAccuracy = 10000;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};

  FILE* f = fopen("pytorch/vision/test/assets/videos/R6llTwEh07w.mp4", "rb");
  CHECK(f != nullptr);
  fseek(f, 0, SEEK_END);
  std::vector<uint8_t> buffer(ftell(f));
  rewind(f);
  CHECK_EQ(buffer.size(), fread(buffer.data(), 1, buffer.size(), f));
  fclose(f);

  params.maxSeekableBytes = buffer.size() + 1;
  MemoryBuffer object(buffer.data(), buffer.size());
  CHECK(decoder.init(
      params,
      [object](uint8_t* out, int size, int whence, uint64_t timeoutMs) mutable
      -> int {
        if (out) { // see defs.h file
          // read mode
          return object.read(out, size);
        }
        // seek mode
        if (!timeoutMs) {
          // seek capabilty, yes - no
          return -1;
        }
        return object.seek(size, whence);
      }));
  DecoderOutputMessage out;
  while (0 == decoder.decode(&out, 100)) {
    LOG(INFO) << "Decoded frame, timestamp(us): " << out.header.pts
              << ", num: " << out.header.format.num
              << ", den: " << out.header.format.den
              << ", duration(us): " << out.header.format.duration;
  }
  decoder.shutdown();
}

TEST(SyncDecoder, TestMemoryBufferNoSeekableWithPartialRead) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.endOffset = 9000000;
  params.seekAccuracy = 10000;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};

  FILE* f = fopen("pytorch/vision/test/assets/videos/R6llTwEh07w.mp4", "rb");
  CHECK(f != nullptr);
  fseek(f, 0, SEEK_END);
  std::vector<uint8_t> buffer(ftell(f));
  rewind(f);
  CHECK_EQ(buffer.size(), fread(buffer.data(), 1, buffer.size(), f));
  fclose(f);

  params.maxSeekableBytes = buffer.size() / 2;
  MemoryBuffer object(buffer.data(), buffer.size());
  CHECK(!decoder.init(
      params,
      [object](uint8_t* out, int size, int whence, uint64_t timeoutMs) mutable
      -> int {
        if (out) { // see defs.h file
          // read mode
          return object.read(out, size);
        }
        // seek mode
        if (!timeoutMs) {
          // seek capabilty, yes - no
          return -1;
        }
        return object.seek(size, whence);
      }));
}
