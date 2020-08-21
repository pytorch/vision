
# include "Video.h"
#include <torch/script.h>
#include <c10/util/Logging.h>
#include "sync_decoder.h"
#include "sync_decoder.h"
#include "memory_buffer.h"
#include "defs.h"


using namespace std;
using namespace ffmpeg;


// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_video_reader(void) {
  // No need to do anything.
  return NULL;
}
#else
PyMODINIT_FUNC PyInit_video_reader(void) {
  // No need to do anything.
  return NULL;
}
#endif
#endif


const size_t decoderTimeoutMs = 600000;
const AVPixelFormat defaultVideoPixelFormat = AV_PIX_FMT_RGB24;
const AVSampleFormat defaultAudioSampleFormat = AV_SAMPLE_FMT_FLT;
// A jitter can be added to the end of the range to avoid conversion/rounding
// error, small value 100us won't be enough to select the next frame, but enough
// to compensate rounding error due to the multiple conversions.
const size_t timeBaseJitterUs = 100;


std::string parse_type_to_string(const std::string& stream_string) {
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
    return device->first;
  }
  AT_ERROR(
      "Expected one of [audio, video, subtitle, cc] ", stream_string);
}

MediaType parse_type_to_mt(const std::string& stream_string) {
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
    return device->second;
  }
  AT_ERROR(
      "Expected one of [audio, video, subtitle, cc] ", stream_string);
}

std::tuple<std::string, int64_t> Video::_parseStream(const std::string& streamString){
    TORCH_CHECK(!streamString.empty(), "Stream string must not be empty");
    static const std::regex regex("([a-zA-Z_]+)(?::([1-9]\\d*|0))?");
    std::smatch match;

    TORCH_CHECK(
        std::regex_match(streamString, match, regex),
        "Invalid stream string: '", streamString, "'");
    
    std::string type_ = "video";
    type_ = parse_type_to_string(match[1].str());
    int64_t index_ = -1;
    if (match[2].matched) {
        try {
        index_ = c10::stoi(match[2].str());
        } catch (const std::exception &) {
        AT_ERROR(
            "Could not parse device index '", match[2].str(),
            "' in device string '", streamString, "'");
        }
    }
    return std::make_tuple(type_, index_);
}


void Video::_getDecoderParams(
        int64_t videoStartUs,
        int64_t getPtsOnly,
        std::string stream,
        long stream_id=-1,
        bool all_streams=false,
        double seekFrameMarginUs=10){

    params.headerOnly = getPtsOnly != 0;
    params.seekAccuracy = seekFrameMarginUs;
    params.startOffset = videoStartUs;
    params.timeoutMs = decoderTimeoutMs;
    params.preventStaleness = false;  // not sure what this is about

    if (all_streams == true){
        MediaFormat audioFormat((long) -2);
        audioFormat.type = TYPE_AUDIO;
        audioFormat.format.audio.format = defaultAudioSampleFormat;
        params.formats.insert(audioFormat);

        MediaFormat videoFormat(0, (long) -2);
        videoFormat.type = TYPE_VIDEO;
        videoFormat.format.video.format = defaultVideoPixelFormat;
        params.formats.insert(videoFormat);

        // there is no clear way on how to use other formats- todo later
        MediaFormat subtitleFormat(char('0'), long(-2));
        subtitleFormat.type = TYPE_SUBTITLE;
        params.formats.insert(subtitleFormat);

        MediaFormat ccFormat(double(0), long(-2));
        ccFormat.type = TYPE_CC;
        params.formats.insert(ccFormat);

    }

    // else use the stream using the correct parsing technique

} // _get decoder params


Video::Video(
    std::string videoPath, 
    std::string stream, 
    bool isReadFile) {

    //parse stream information
    current_stream = _parseStream(stream);
    // note that in the initial version we want to get all streams

    Video::_getDecoderParams(
        0,      // video start
        false,  //headerOnly
        get<0>(current_stream),
        long(-2),     // stream_id parsed from info above
        true    // read all streams
    );

    std::string logMessage, logType;
    DecoderInCallback callback = nullptr;
    // TODO: add read from memory option
    params.uri = videoPath;
    logType = "file";
    logMessage = videoPath;
    

    // get a decoder
    bool succeeded;

    cout << "Video decoding to gather metadata from " << logType << " [" << logMessage
          << "] has started";
    
    std::vector<double> videoFPS, audioFPS, ccFPS, subtitleFPS;

    std::vector<DecoderMetadata> metadata;
    succeeded = decoder.init(params, std::move(callback), &metadata);
    if (succeeded) {
        for (const auto& header : metadata) {
            cout << "Decoding stream of" << header.format.type ;
        
            // generate streamMetadata object
            // std::map<std::string, double> streamInfo;
            // parse stream timebase
            // streamInfo.insert({"timeBase", (double) (header.num / header.den)});
            // parse stream duration
            // to get duration in seconds multiply duration by timebase
            // streamInfo.insert({"duration", (double) header.duration * (double) (header.num / header.den)});
                        
            if (header.format.type == TYPE_VIDEO) {
                // parse stream fps
                double fps = double(header.fps);
                videoFPS.push_back(fps);
            } else if (header.format.type == TYPE_AUDIO) {
                // parse stream fps (user defined, doesn't seem cool)
                double fps = double(0);
                audioFPS.push_back(fps);
            } else{
                cout << "Got type" << header.format.type; 
            };
        }

    } else{
        audioFPS.push_back((-1.0));
        videoFPS.push_back((-1.0));

    }
    streamMetadata.insert({"video", videoFPS});
    streamMetadata.insert({"audio", audioFPS});
} //video

std::tuple<std::string, int64_t> Video::getCurrentStream() const {
    return current_stream;
}

std::vector<double> Video::getFPS(std::string stream) const{
    // add safety check
    std::string stream_str = parse_type_to_string(stream);
    return streamMetadata.at(stream_str);
}


// std::map<std::string, std::vector<std::map<std::string, double>>> Video::getMetadata() const {
//     return VideoMetadata;
// }


