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

 public:
  Video(std::string videoPath, std::string stream);
  std::tuple<std::string, int64_t> getCurrentStream() const;
  c10::Dict<std::string, c10::Dict<std::string, std::vector<double>>>
  getStreamMetadata() const;
  void Seek(double ts);
  bool setCurrentStream(std::string stream);
  std::tuple<torch::Tensor, double> Next();

 private:
  bool succeeded = false; // decoder init flag
  // seekTS and doSeek act as a flag - if it's not set, next function simply
  // retruns the next frame. If it's set, we look at the global seek
  // time in comination with any_frame settings
  double seekTS = -1;

  void _getDecoderParams(
      double videoStartS,
      int64_t getPtsOnly,
      std::string stream,
      long stream_id,
      bool all_streams,
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
