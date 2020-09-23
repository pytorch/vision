
#include "Video.h"
#include <c10/util/Logging.h>
#include <torch/script.h>
#include "defs.h"
#include "memory_buffer.h"
#include "sync_decoder.h"

using namespace std;
using namespace ffmpeg;

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
// #ifdef _WIN32
// #if PY_MAJOR_VERSION < 3
// PyMODINIT_FUNC init_video_reader(void) {
//   // No need to do anything.
//   return NULL;
// }
// #else
// PyMODINIT_FUNC PyInit_video_reader(void) {
//   // No need to do anything.
//   return NULL;
// }
// #endif
// #endif


const size_t decoderTimeoutMs = 600000;
const AVSampleFormat defaultAudioSampleFormat = AV_SAMPLE_FMT_FLT;
// A jitter can be added to the end of the range to avoid conversion/rounding
// error, small value 100us won't be enough to select the next frame, but enough
// to compensate rounding error due to the multiple conversions.
const size_t timeBaseJitterUs = 100;

// returns number of written bytes
template <typename T>
size_t fillTensorList(
    DecoderOutputMessage& msgs,
    torch::Tensor& frame,
    torch::Tensor& framePts) {
  // set up PTS data
  const auto& msg = msgs;

  float* framePtsData = framePts.data_ptr<float>();

  float pts_s = float(float(msg.header.pts) * 1e-6);
  framePtsData[0] = pts_s;

  T* frameData = frame.numel() > 0 ? frame.data_ptr<T>() : nullptr;

  if (frameData) {
    auto sizeInBytes = msg.payload->length();
    memcpy(frameData, msg.payload->data(), sizeInBytes);
  }
  return sizeof(T);
}

size_t fillVideoTensor(
    DecoderOutputMessage& msgs,
    torch::Tensor& videoFrame,
    torch::Tensor& videoFramePts) {
  return fillTensorList<uint8_t>(msgs, videoFrame, videoFramePts);
}

size_t fillAudioTensor(
    DecoderOutputMessage& msgs,
    torch::Tensor& audioFrame,
    torch::Tensor& audioFramePts) {
  return fillTensorList<float>(msgs, audioFrame, audioFramePts);
}

std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, ffmpeg::MediaType> const* _parse_type(const std::string& stream_string) {
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
  AT_ERROR("Expected one of [audio, video, subtitle, cc] ", stream_string);
}

std::string parse_type_to_string(const std::string& stream_string){
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
      AT_ERROR(
          "Could not parse device index '",
          match[2].str(),
          "' in device string '",
          streamString,
          "'");
    }
  }
  return std::make_tuple(type_, index_);
}

void Video::_getDecoderParams(
    double videoStartS,
    int64_t getPtsOnly,
    std::string stream,
    long stream_id = -1,
    bool all_streams = false,
    double seekFrameMarginUs = 10) {
  
  int64_t videoStartUs = int64_t(videoStartS * 1e6);

  params.timeoutMs = decoderTimeoutMs;
  params.startOffset = videoStartUs;
  params.seekAccuracy = 10;
  params.headerOnly = false;

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
      // format.format.video.format = defaultVideoPixelFormat;
    }
    params.formats.insert(format);
  }

} // _get decoder params

Video::Video(std::string videoPath, std::string stream, bool isReadFile) {
  // parse stream information
  current_stream = _parseStream(stream);
  // note that in the initial call we want to get all streams
  Video::_getDecoderParams(
      0, // video start
      0, // headerOnly
      get<0>(current_stream), // stream info - remove that
      long(-1), // stream_id parsed from info above change to -2
      true // read all streams
  );

  std::string logMessage, logType;

  // TODO: add read from memory option
  params.uri = videoPath;
  logType = "file";
  logMessage = videoPath;

  std::vector<double> audioFPS, videoFPS, ccFPS, subsFPS;
  std::vector<double> audioDuration, videoDuration, ccDuration, subsDuration;
  std::vector<double> audioTB, videoTB, ccTB, subsTB;

  // calback and metadata defined in struct
  succeeded = decoder.init(params, std::move(callback), &metadata);
  if (succeeded) {
    for (const auto& header : metadata) {
      double fps = double(header.fps);
      double timeBase = double(header.num) / double(header.den);
      double duration = double(header.duration) * 1e-6; // * timeBase;

      if (header.format.type == TYPE_VIDEO) {
        videoFPS.push_back(fps);
        videoDuration.push_back(duration);
      } else if (header.format.type == TYPE_AUDIO) {
        audioFPS.push_back(fps);
        audioDuration.push_back(duration);
      } else if (header.format.type == TYPE_CC) {
        ccFPS.push_back(fps);
        ccDuration.push_back(duration);
      } else if (header.format.type == TYPE_SUBTITLE) {
        subsFPS.push_back(fps);
        subsDuration.push_back(duration);
      };
    }
  }
  streamFPS.insert({{"video", videoFPS}, {"audio", audioFPS}});
  streamDuration.insert({{"video", videoDuration}, {"audio", audioDuration}});

  succeeded = Video::_setCurrentStream();
  LOG(INFO) << "\nDecoder inited with: " << succeeded << "\n";
  if (get<1>(current_stream) != -1) {
    LOG(INFO)
        << "Stream index set to " << get<1>(current_stream)
        << ". If you encounter trouble, consider switching it to automatic stream discovery. \n";
  }
} // video

bool Video::_setCurrentStream() {
  double ts = 0;
  if (seekTS > 0) {
    ts = seekTS;
  }

  _getDecoderParams(
      ts, // video start
      0, // headerOnly
      get<0>(current_stream), // stream
      long(get<1>(
          current_stream)), // stream_id parsed from info above change to -2
      false // read all streams
  );

  // calback and metadata defined in Video.h
  return (decoder.init(params, std::move(callback), &metadata));
}

std::tuple<std::string, int64_t> Video::getCurrentStream() const {
  return current_stream;
}

std::vector<double> Video::getFPS(std::string stream) const {
  // add safety check
  if (stream.empty()) {
    stream = get<0>(current_stream);
  }
  auto stream_tpl = _parseStream(stream);
  std::string stream_str = get<0>(stream_tpl);
  // check if the stream exists
  return streamFPS.at(stream_str);
}

std::vector<double> Video::getDuration(std::string stream) const {
  // add safety check
  if (stream.empty()) {
    stream = get<0>(current_stream);
  }
  auto stream_tpl = _parseStream(stream);
  std::string stream_str = get<0>(stream_tpl);
  // check if the stream exists
  return streamDuration.at(stream_str);
}

void Video::Seek(double ts, bool any_frame = false) {
  // initialize the class variables used for seeking and retrurn
  video_any_frame = any_frame;
  seekTS = ts;
  doSeek = true;
}

torch::List<torch::Tensor> Video::Next(std::string stream) {
  
  bool newInit = false; // avoid unnecessary decoder initializations
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
      current_stream = _parseStream(stream);
      newInit = true;
  }

  if ((seekTS != -1) && (doSeek == true)) {
      newInit = true;
      doSeek = false;
  }

  if (newInit){
    succeeded = Video::_setCurrentStream();
    if (succeeded) {
      newInit = false;
    }
  }

  // if failing to decode simply return a null tensor (note, should we
  // raise an exeption?)
  torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);
  torch::Tensor outFrame = torch::zeros({0}, torch::kByte);

  // decode single frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  // if successfull
  if (res == 0) {
    auto header = out.header;
    const auto& format = header.format;

    // initialize the output variables based on type
    size_t expectedWrittenBytes = 0;

    if (format.type == TYPE_VIDEO) {
      // note: this can potentially be optimized
      // by having the global tensor that we fill at decode time
      // (would avoid allocations)
      int outHeight = format.format.video.height;
      int outWidth = format.format.video.width;
      int numChannels = 3;
      outFrame = torch::zeros({outHeight, outWidth, numChannels}, torch::kByte);
      expectedWrittenBytes = outHeight * outWidth * numChannels;
    } else if (format.type == TYPE_AUDIO) {
      int outAudioChannels = format.format.audio.channels;
      int bytesPerSample = av_get_bytes_per_sample(
          static_cast<AVSampleFormat>(format.format.audio.format));
      int frameSizeTotal = out.payload->length();

      CHECK_EQ(frameSizeTotal % (outAudioChannels * bytesPerSample), 0);
      int numAudioSamples =
          frameSizeTotal / (outAudioChannels * bytesPerSample);

      outFrame =
          torch::zeros({numAudioSamples, outAudioChannels}, torch::kFloat);

      expectedWrittenBytes = numAudioSamples * outAudioChannels * sizeof(float);
    }
    // currently not supporting other formats (will do soon)

    // note: this will need to be revised to support less-accurate seek. So far keep as is
    if (format.type == TYPE_VIDEO) {
      auto numberWrittenBytes = fillVideoTensor(out, outFrame, framePTS);
    } else {
      auto numberWrittenBytes = fillAudioTensor(out, outFrame, framePTS);
    }
    out.payload.reset();
  } else{
    LOG(ERROR) << "Decoder failed ( or ran into last iteration)";
  }

  torch::List<torch::Tensor> result;
  result.push_back(outFrame);
  result.push_back(framePTS);
  return result;
}

// Video::~Video() {
  // destructor to be defined thoroughly later
//   delete params; // does not have destructor
//   delete metadata; // struct does not have destructor
//   delete decoder; // should be fine
//   delete streamFPS; // should be fine
//   delete streamDuration; // should be fine
// }
