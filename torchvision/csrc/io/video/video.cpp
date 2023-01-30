#include "video.h"

#include <regex>

namespace vision {
namespace video {

namespace {

const size_t decoderTimeoutMs = 600000;
const AVPixelFormat defaultVideoPixelFormat = AV_PIX_FMT_RGB24;

// returns number of written bytes
template <typename T>
size_t fillTensorList(DecoderOutputMessage& msgs, torch::Tensor& frame) {
  const auto& msg = msgs;
  T* frameData = frame.numel() > 0 ? frame.data_ptr<T>() : nullptr;
  if (frameData) {
    auto sizeInBytes = msg.payload->length();
    memcpy(frameData, msg.payload->data(), sizeInBytes);
  }
  return sizeof(T);
}

size_t fillVideoTensor(DecoderOutputMessage& msgs, torch::Tensor& videoFrame) {
  return fillTensorList<uint8_t>(msgs, videoFrame);
}

size_t fillAudioTensor(DecoderOutputMessage& msgs, torch::Tensor& audioFrame) {
  return fillTensorList<float>(msgs, audioFrame);
}

std::array<std::pair<std::string, ffmpeg::MediaType>, 4>::const_iterator
_parse_type(const std::string& stream_string) {
  static const std::array<std::pair<std::string, MediaType>, 4> types = {{
      {"video", TYPE_VIDEO},
      {"audio", TYPE_AUDIO},
      {"subtitle", TYPE_SUBTITLE},
      {"cc", TYPE_CC},
  }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [stream_string](const std::pair<std::string, MediaType>& p) {
        return p.first == stream_string;
      });
  if (device != types.end()) {
    return device;
  }
  TORCH_CHECK(
      false, "Expected one of [audio, video, subtitle, cc] ", stream_string);
}

std::string parse_type_to_string(const std::string& stream_string) {
  auto device = _parse_type(stream_string);
  return device->first;
}

MediaType parse_type_to_mt(const std::string& stream_string) {
  auto device = _parse_type(stream_string);
  return device->second;
}

std::tuple<std::string, long> _parseStream(const std::string& streamString) {
  TORCH_CHECK(!streamString.empty(), "Stream string must not be empty");
  static const std::regex regex("([a-zA-Z_]+)(?::([1-9]\\d*|0))?");
  std::smatch match;

  TORCH_CHECK(
      std::regex_match(streamString, match, regex),
      "Invalid stream string: '",
      streamString,
      "'");

  std::string type_ = "video";
  type_ = parse_type_to_string(match[1].str());
  long index_ = -1;
  if (match[2].matched) {
    try {
      index_ = c10::stoi(match[2].str());
    } catch (const std::exception&) {
      TORCH_CHECK(
          false,
          "Could not parse device index '",
          match[2].str(),
          "' in device string '",
          streamString,
          "'");
    }
  }
  return std::make_tuple(type_, index_);
}

} // namespace

void Video::_getDecoderParams(
    double videoStartS,
    int64_t getPtsOnly,
    std::string stream,
    long stream_id = -1,
    bool fastSeek = true,
    bool all_streams = false,
    int64_t num_threads = 1,
    double seekFrameMarginUs = 10) {
  int64_t videoStartUs = int64_t(videoStartS * 1e6);

  params.timeoutMs = decoderTimeoutMs;
  params.startOffset = videoStartUs;
  params.seekAccuracy = seekFrameMarginUs;
  params.fastSeek = fastSeek;
  params.headerOnly = false;
  params.numThreads = num_threads;

  params.preventStaleness = false; // not sure what this is about

  if (all_streams == true) {
    MediaFormat format;
    format.stream = -2;
    format.type = TYPE_AUDIO;
    params.formats.insert(format);

    format.type = TYPE_VIDEO;
    format.stream = -2;
    format.format.video.width = 0;
    format.format.video.height = 0;
    format.format.video.cropImage = 0;
    format.format.video.format = defaultVideoPixelFormat;
    params.formats.insert(format);

    format.type = TYPE_SUBTITLE;
    format.stream = -2;
    params.formats.insert(format);

    format.type = TYPE_CC;
    format.stream = -2;
    params.formats.insert(format);
  } else {
    // parse stream type
    MediaType stream_type = parse_type_to_mt(stream);

    // TODO: reset params.formats
    std::set<MediaFormat> formats;
    params.formats = formats;
    // Define new format
    MediaFormat format;
    format.type = stream_type;
    format.stream = stream_id;
    if (stream_type == TYPE_VIDEO) {
      format.format.video.width = 0;
      format.format.video.height = 0;
      format.format.video.cropImage = 0;
      format.format.video.format = defaultVideoPixelFormat;
    }
    params.formats.insert(format);
  }

} // _get decoder params

void Video::initFromFile(
    std::string videoPath,
    std::string stream,
    int64_t numThreads) {
  TORCH_CHECK(!initialized, "Video object can only be initialized once");
  initialized = true;
  params.uri = videoPath;
  _init(stream, numThreads);
}

void Video::initFromMemory(
    torch::Tensor videoTensor,
    std::string stream,
    int64_t numThreads) {
  TORCH_CHECK(!initialized, "Video object can only be initialized once");
  initialized = true;
  callback = MemoryBuffer::getCallback(
      videoTensor.data_ptr<uint8_t>(), videoTensor.size(0));
  _init(stream, numThreads);
}

void Video::_init(std::string stream, int64_t numThreads) {
  // set number of threads global
  numThreads_ = numThreads;
  // parse stream information
  current_stream = _parseStream(stream);
  // note that in the initial call we want to get all streams
  _getDecoderParams(
      0, // video start
      0, // headerOnly
      std::get<0>(current_stream), // stream info - remove that
      long(-1), // stream_id parsed from info above change to -2
      false, // fastseek: we're using the default param here
      true, // read all streams
      numThreads_ // global number of Threads for decoding
  );

  std::string logMessage, logType;

  // locals
  std::vector<double> audioFPS, videoFPS;
  std::vector<double> audioDuration, videoDuration, ccDuration, subsDuration;
  std::vector<double> audioTB, videoTB, ccTB, subsTB;
  c10::Dict<std::string, std::vector<double>> audioMetadata;
  c10::Dict<std::string, std::vector<double>> videoMetadata;
  c10::Dict<std::string, std::vector<double>> ccMetadata;
  c10::Dict<std::string, std::vector<double>> subsMetadata;

  // callback and metadata defined in struct
  DecoderInCallback tmp_callback = callback;
  succeeded = decoder.init(params, std::move(tmp_callback), &metadata);
  if (succeeded) {
    for (const auto& header : metadata) {
      double fps = double(header.fps);
      double duration = double(header.duration) * 1e-6; // * timeBase;

      if (header.format.type == TYPE_VIDEO) {
        videoFPS.push_back(fps);
        videoDuration.push_back(duration);
      } else if (header.format.type == TYPE_AUDIO) {
        audioFPS.push_back(fps);
        audioDuration.push_back(duration);
      } else if (header.format.type == TYPE_CC) {
        ccDuration.push_back(duration);
      } else if (header.format.type == TYPE_SUBTITLE) {
        subsDuration.push_back(duration);
      };
    }
  }
  // audio
  audioMetadata.insert("duration", audioDuration);
  audioMetadata.insert("framerate", audioFPS);
  // video
  videoMetadata.insert("duration", videoDuration);
  videoMetadata.insert("fps", videoFPS);
  // subs
  subsMetadata.insert("duration", subsDuration);
  // cc
  ccMetadata.insert("duration", ccDuration);
  // put all to a data
  streamsMetadata.insert("video", videoMetadata);
  streamsMetadata.insert("audio", audioMetadata);
  streamsMetadata.insert("subtitles", subsMetadata);
  streamsMetadata.insert("cc", ccMetadata);

  succeeded = setCurrentStream(stream);
  LOG(INFO) << "\nDecoder inited with: " << succeeded << "\n";
  if (std::get<1>(current_stream) != -1) {
    LOG(INFO)
        << "Stream index set to " << std::get<1>(current_stream)
        << ". If you encounter trouble, consider switching it to automatic stream discovery. \n";
  }
}

Video::Video(std::string videoPath, std::string stream, int64_t numThreads) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.io.video.video.Video");
  if (!videoPath.empty()) {
    initFromFile(videoPath, stream, numThreads);
  }
} // video

bool Video::setCurrentStream(std::string stream = "video") {
  TORCH_CHECK(initialized, "Video object has to be initialized first");
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
    current_stream = _parseStream(stream);
  }

  double ts = 0;
  if (seekTS > 0) {
    ts = seekTS;
  }

  _getDecoderParams(
      ts, // video start
      0, // headerOnly
      std::get<0>(current_stream), // stream
      long(std::get<1>(
          current_stream)), // stream_id parsed from info above change to -2
      false, // fastseek param set to 0 false by default (changed in seek)
      false, // read all streams
      numThreads_ // global number of threads
  );

  // callback and metadata defined in Video.h
  DecoderInCallback tmp_callback = callback;
  return (decoder.init(params, std::move(tmp_callback), &metadata));
}

std::tuple<std::string, int64_t> Video::getCurrentStream() const {
  TORCH_CHECK(initialized, "Video object has to be initialized first");
  return current_stream;
}

c10::Dict<std::string, c10::Dict<std::string, std::vector<double>>> Video::
    getStreamMetadata() const {
  TORCH_CHECK(initialized, "Video object has to be initialized first");
  return streamsMetadata;
}

void Video::Seek(double ts, bool fastSeek = false) {
  TORCH_CHECK(initialized, "Video object has to be initialized first");
  // initialize the class variables used for seeking and retrurn
  _getDecoderParams(
      ts, // video start
      0, // headerOnly
      std::get<0>(current_stream), // stream
      long(std::get<1>(
          current_stream)), // stream_id parsed from info above change to -2
      fastSeek, // fastseek
      false, // read all streams
      numThreads_ // global number of threads
  );

  // callback and metadata defined in Video.h
  DecoderInCallback tmp_callback = callback;
  succeeded = decoder.init(params, std::move(tmp_callback), &metadata);

  LOG(INFO) << "Decoder init at seek " << succeeded << "\n";
}

std::tuple<torch::Tensor, double> Video::Next() {
  TORCH_CHECK(initialized, "Video object has to be initialized first");
  // if failing to decode simply return a null tensor (note, should we
  // raise an exception?)
  double frame_pts_s;
  torch::Tensor outFrame = torch::zeros({0}, torch::kByte);

  // decode single frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  // if successful
  if (res == 0) {
    frame_pts_s = double(double(out.header.pts) * 1e-6);

    auto header = out.header;
    const auto& format = header.format;

    // initialize the output variables based on type

    if (format.type == TYPE_VIDEO) {
      // note: this can potentially be optimized
      // by having the global tensor that we fill at decode time
      // (would avoid allocations)
      int outHeight = format.format.video.height;
      int outWidth = format.format.video.width;
      int numChannels = 3;
      outFrame = torch::zeros({outHeight, outWidth, numChannels}, torch::kByte);
      fillVideoTensor(out, outFrame);
      outFrame = outFrame.permute({2, 0, 1});

    } else if (format.type == TYPE_AUDIO) {
      int outAudioChannels = format.format.audio.channels;
      int bytesPerSample = av_get_bytes_per_sample(
          static_cast<AVSampleFormat>(format.format.audio.format));
      int frameSizeTotal = out.payload->length();

      TORCH_CHECK_EQ(frameSizeTotal % (outAudioChannels * bytesPerSample), 0);
      int numAudioSamples =
          frameSizeTotal / (outAudioChannels * bytesPerSample);

      outFrame =
          torch::zeros({numAudioSamples, outAudioChannels}, torch::kFloat);

      fillAudioTensor(out, outFrame);
    }
    // currently not supporting other formats (will do soon)

    out.payload.reset();
  } else if (res == ENODATA) {
    LOG(INFO) << "Decoder ran out of frames (ENODATA)\n";
  } else {
    LOG(ERROR) << "Decoder failed with ERROR_CODE " << res;
  }

  return std::make_tuple(outFrame, frame_pts_s);
}

static auto registerVideo =
    torch::class_<Video>("torchvision", "Video")
        .def(torch::init<std::string, std::string, int64_t>())
        .def("init_from_file", &Video::initFromFile)
        .def("init_from_memory", &Video::initFromMemory)
        .def("get_current_stream", &Video::getCurrentStream)
        .def("set_current_stream", &Video::setCurrentStream)
        .def("get_metadata", &Video::getStreamMetadata)
        .def("seek", &Video::Seek)
        .def("next", &Video::Next);

} // namespace video
} // namespace vision
