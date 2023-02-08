#include <c10/util/Logging.h>
#include <dirent.h>
#include <gtest/gtest.h>
#include "memory_buffer.h"
#include "sync_decoder.h"
#include "util.h"

using namespace ffmpeg;

namespace {
struct VideoFileStats {
  std::string name;
  size_t durationPts{0};
  int num{0};
  int den{0};
  int fps{0};
};

void gotAllTestFiles(
    const std::string& folder,
    std::vector<VideoFileStats>* stats) {
  DIR* d = opendir(folder.c_str());
  CHECK(d);
  struct dirent* dir;
  while ((dir = readdir(d))) {
    if (dir->d_type != DT_DIR && 0 != strcmp(dir->d_name, "README")) {
      VideoFileStats item;
      item.name = folder + '/' + dir->d_name;
      LOG(INFO) << "Found video file: " << item.name;
      stats->push_back(std::move(item));
    }
  }
  closedir(d);
}

void gotFilesStats(std::vector<VideoFileStats>& stats) {
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.seekAccuracy = 100000;
  params.formats = {MediaFormat(0)};
  params.headerOnly = true;
  params.preventStaleness = false;
  size_t avgProvUs = 0;
  const size_t rounds = 100;
  for (auto& item : stats) {
    LOG(INFO) << "Decoding video file in memory: " << item.name;
    FILE* f = fopen(item.name.c_str(), "rb");
    CHECK(f != nullptr);
    fseek(f, 0, SEEK_END);
    std::vector<uint8_t> buffer(ftell(f));
    rewind(f);
    size_t s = fread(buffer.data(), 1, buffer.size(), f);
    TORCH_CHECK_EQ(buffer.size(), s);
    fclose(f);

    for (size_t i = 0; i < rounds; ++i) {
      SyncDecoder decoder;
      std::vector<DecoderMetadata> metadata;
      const auto now = std::chrono::steady_clock::now();
      CHECK(decoder.init(
          params,
          MemoryBuffer::getCallback(buffer.data(), buffer.size()),
          &metadata));
      const auto then = std::chrono::steady_clock::now();
      decoder.shutdown();
      avgProvUs +=
          std::chrono::duration_cast<std::chrono::microseconds>(then - now)
              .count();
      TORCH_CHECK_EQ(metadata.size(), 1);
      item.num = metadata[0].num;
      item.den = metadata[0].den;
      item.fps = metadata[0].fps;
      item.durationPts =
          av_rescale_q(metadata[0].duration, AV_TIME_BASE_Q, {1, item.fps});
    }
  }
  LOG(INFO) << "Probing (us) " << avgProvUs / stats.size() / rounds;
}

size_t measurePerformanceUs(
    const std::vector<VideoFileStats>& stats,
    size_t rounds,
    size_t num,
    size_t stride) {
  size_t avgClipDecodingUs = 0;
  std::srand(time(nullptr));
  for (const auto& item : stats) {
    FILE* f = fopen(item.name.c_str(), "rb");
    CHECK(f != nullptr);
    fseek(f, 0, SEEK_END);
    std::vector<uint8_t> buffer(ftell(f));
    rewind(f);
    size_t s = fread(buffer.data(), 1, buffer.size(), f);
    TORCH_CHECK_EQ(buffer.size(), s);
    fclose(f);

    for (size_t i = 0; i < rounds; ++i) {
      // randomy select clip
      size_t rOffset = std::rand();
      size_t fOffset = rOffset % item.durationPts;
      size_t clipFrames = num + (num - 1) * stride;
      if (fOffset + clipFrames > item.durationPts) {
        fOffset = item.durationPts - clipFrames;
      }

      DecoderParameters params;
      params.timeoutMs = 10000;
      params.startOffset = 1000000;
      params.seekAccuracy = 100000;
      params.preventStaleness = false;

      for (size_t n = 0; n < num; ++n) {
        std::list<DecoderOutputMessage> msgs;

        params.startOffset =
            av_rescale_q(fOffset, {1, item.fps}, AV_TIME_BASE_Q);
        params.endOffset = params.startOffset + 100;

        auto now = std::chrono::steady_clock::now();
        SyncDecoder decoder;
        CHECK(decoder.init(
            params,
            MemoryBuffer::getCallback(buffer.data(), buffer.size()),
            nullptr));
        DecoderOutputMessage out;
        while (0 == decoder.decode(&out, params.timeoutMs)) {
          msgs.push_back(std::move(out));
        }

        decoder.shutdown();

        const auto then = std::chrono::steady_clock::now();

        fOffset += 1 + stride;

        avgClipDecodingUs +=
            std::chrono::duration_cast<std::chrono::microseconds>(then - now)
                .count();
      }
    }
  }

  return avgClipDecodingUs / rounds / num / stats.size();
}

void runDecoder(SyncDecoder& decoder) {
  DecoderOutputMessage out;
  size_t audioFrames = 0, videoFrames = 0, totalBytes = 0;
  while (0 == decoder.decode(&out, 10000)) {
    if (out.header.format.type == TYPE_AUDIO) {
      ++audioFrames;
    } else if (out.header.format.type == TYPE_VIDEO) {
      ++videoFrames;
    } else if (out.header.format.type == TYPE_SUBTITLE && out.payload) {
      // deserialize
      LOG(INFO) << "Deserializing subtitle";
      AVSubtitle sub;
      memset(&sub, 0, sizeof(sub));
      EXPECT_TRUE(Util::deserialize(*out.payload, &sub));
      LOG(INFO) << "Found subtitles"
                << ", num rects: " << sub.num_rects;
      for (int i = 0; i < sub.num_rects; ++i) {
        std::string text = "picture";
        if (sub.rects[i]->type == SUBTITLE_TEXT) {
          text = sub.rects[i]->text;
        } else if (sub.rects[i]->type == SUBTITLE_ASS) {
          text = sub.rects[i]->ass;
        }

        LOG(INFO) << "Rect num: " << i << ", type:" << sub.rects[i]->type
                  << ", text: " << text;
      }

      avsubtitle_free(&sub);
    }
    if (out.payload) {
      totalBytes += out.payload->length();
    }
  }
  LOG(INFO) << "Decoded audio frames: " << audioFrames
            << ", video frames: " << videoFrames
            << ", total bytes: " << totalBytes;
}
} // namespace

TEST(SyncDecoder, TestSyncDecoderPerformance) {
  // Measure the average time of decoding per clip
  // 1. list of the videos in testing directory
  // 2. for each video got number of frames with timestamps
  // 3. randomly select frame offset
  // 4. adjust offset for number frames and strides,
  //    if it's out out upper boundary
  // 5. repeat multiple times, measuring and accumulating decoding time
  //    per clip.
  /*
  1) 4 x 2
  2) 8 x 8
  3) 16 x 8
  4) 32 x 4
  */
  const std::string kFolder = "pytorch/vision/test/assets/videos";
  std::vector<VideoFileStats> stats;
  gotAllTestFiles(kFolder, &stats);
  gotFilesStats(stats);

  const size_t kRounds = 10;

  auto new4x2 = measurePerformanceUs(stats, kRounds, 4, 2);
  auto new8x8 = measurePerformanceUs(stats, kRounds, 8, 8);
  auto new16x8 = measurePerformanceUs(stats, kRounds, 16, 8);
  auto new32x4 = measurePerformanceUs(stats, kRounds, 32, 4);
  LOG(INFO) << "Clip decoding (us)"
            << ", new(4x2): " << new4x2 << ", new(8x8): " << new8x8
            << ", new(16x8): " << new16x8 << ", new(32x4): " << new32x4;
}

TEST(SyncDecoder, Test) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.seekAccuracy = 100000;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};
  params.uri = "pytorch/vision/test/assets/videos/R6llTwEh07w.mp4";
  CHECK(decoder.init(params, nullptr, nullptr));
  runDecoder(decoder);
  decoder.shutdown();
}

TEST(SyncDecoder, TestSubtitles) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};
  params.uri = "vue/synergy/data/robotsub.mp4";
  CHECK(decoder.init(params, nullptr, nullptr));
  runDecoder(decoder);
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
  CHECK(decoder.init(params, nullptr, nullptr));
  runDecoder(decoder);
  decoder.shutdown();

  params.uri = "pytorch/vision/test/assets/videos/SOX5yA1l24A.mp4";
  CHECK(decoder.init(params, nullptr, nullptr));
  runDecoder(decoder);
  decoder.shutdown();

  params.uri = "pytorch/vision/test/assets/videos/WUzgd7C1pWA.mp4";
  CHECK(decoder.init(params, nullptr, nullptr));
  runDecoder(decoder);
  decoder.shutdown();
}

TEST(SyncDecoder, TestHeadersOnlyDownSampling) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.seekAccuracy = 100000;
  params.headerOnly = true;
  MediaFormat format;
  format.type = TYPE_AUDIO;
  format.format.audio.samples = 8000;
  params.formats.insert(format);

  format.type = TYPE_VIDEO;
  format.format.video.width = 224;
  format.format.video.height = 224;
  params.formats.insert(format);

  params.uri = "pytorch/vision/test/assets/videos/R6llTwEh07w.mp4";
  CHECK(decoder.init(params, nullptr, nullptr));
  runDecoder(decoder);
  decoder.shutdown();

  params.uri = "pytorch/vision/test/assets/videos/SOX5yA1l24A.mp4";
  CHECK(decoder.init(params, nullptr, nullptr));
  runDecoder(decoder);
  decoder.shutdown();

  params.uri = "pytorch/vision/test/assets/videos/WUzgd7C1pWA.mp4";
  CHECK(decoder.init(params, nullptr, nullptr));
  runDecoder(decoder);
  decoder.shutdown();
}

TEST(SyncDecoder, TestInitOnlyNoShutdown) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffset = 1000000;
  params.seekAccuracy = 100000;
  params.headerOnly = false;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};
  params.uri = "pytorch/vision/test/assets/videos/R6llTwEh07w.mp4";
  std::vector<DecoderMetadata> metadata;
  CHECK(decoder.init(params, nullptr, &metadata));
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
  size_t s = fread(buffer.data(), 1, buffer.size(), f);
  TORCH_CHECK_EQ(buffer.size(), s);
  fclose(f);
  CHECK(decoder.init(
      params,
      MemoryBuffer::getCallback(buffer.data(), buffer.size()),
      nullptr));
  LOG(INFO) << "Decoding from memory bytes: " << buffer.size();
  runDecoder(decoder);
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
  size_t s = fread(buffer.data(), 1, buffer.size(), f);
  TORCH_CHECK_EQ(buffer.size(), s);
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
          // seek capability, yes - no
          return -1;
        }
        return object.seek(size, whence);
      },
      nullptr));
  runDecoder(decoder);
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
  size_t s = fread(buffer.data(), 1, buffer.size(), f);
  TORCH_CHECK_EQ(buffer.size(), s);
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
          // seek capability, yes - no
          return -1;
        }
        return object.seek(size, whence);
      },
      nullptr));
}
