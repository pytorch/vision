#pragma once

#include <torch/types.h>

#include "../decoder/defs.h"
#include "../decoder/memory_buffer.h"
#include "../decoder/sync_decoder.h"

using namespace ffmpeg;

namespace vision {
namespace video {

struct Video : torch::CustomClassHolder {
  std::tuple<std::string, long> current_stream; // stream type, id
  // global video metadata
  c10::Dict<std::string, c10::Dict<std::string, std::vector<double>>>
      streamsMetadata;
  int64_t numThreads_{0};

 public:
  Video(
      std::string videoPath = std::string(),
      std::string stream = std::string("video"),
      int64_t numThreads = 0);
  void initFromFile(
      std::string videoPath,
      std::string stream,
      int64_t numThreads);
  void initFromMemory(
      torch::Tensor videoTensor,
      std::string stream,
      int64_t numThreads);

  std::tuple<std::string, int64_t> getCurrentStream() const;
  c10::Dict<std::string, c10::Dict<std::string, std::vector<double>>>
  getStreamMetadata() const;
  void Seek(double ts, bool fastSeek);
  bool setCurrentStream(std::string stream);
  std::tuple<torch::Tensor, double> Next();

 private:
  bool succeeded = false; // decoder init flag
  // seekTS and doSeek act as a flag - if it's not set, next function simply
  // returns the next frame. If it's set, we look at the global seek
  // time in combination with any_frame settings
  double seekTS = -1;

  bool initialized = false;

  void _init(
      std::string stream,
      int64_t numThreads); // expects params.uri OR callback to be set

  void _getDecoderParams(
      double videoStartS,
      int64_t getPtsOnly,
      std::string stream,
      long stream_id,
      bool fastSeek,
      bool all_streams,
      int64_t num_threads,
      double seekFrameMarginUs); // this needs to be improved

  std::map<std::string, std::vector<double>> streamTimeBase; // not used

  DecoderInCallback callback = nullptr;
  std::vector<DecoderMetadata> metadata;

 protected:
  SyncDecoder decoder;
  DecoderParameters params;

}; // struct Video

} // namespace video
} // namespace vision
