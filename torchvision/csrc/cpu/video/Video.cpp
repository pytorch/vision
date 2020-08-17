
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

void Video::_getDecoderParams(
        int64_t videoStartUs,
        int64_t getPtsOnly,
        // how enum works, but stream type
        int stream_id=-1,

        double seekFrameMarginUs=10){

    params.headerOnly = getPtsOnly != 0;
    params.seekAccuracy = seekFrameMarginUs;
    params.startOffset = videoStartUs;
    params.timeoutMs = decoderTimeoutMs;
    params.preventStaleness = false;  // not sure what this is about

} // _get decoder params


Video::Video(
    std::string videoPath, 
    std::string stream, 

    bool isReadFile, 
    int64_t audioSamples=0, 
    int64_t audioChannels=1) {


    //parse stream information

    // set current stream

    DecoderParameters params;
    Video::_getDecoderParams(
        0,      // video start
        false,  //headerOnly
        // stream_type parsed from info above
        -2,     // stream_id parsed from info above
        true    // read all streams
    DecoderParameters params;
    Video::_getDecoderParams(
        0,   // video start
        false,  //headerOnly
        // stream_type parsed from info above
        // stream_id parsed from info above
        audioSamples,
        audioChannels
    );

    std::string logMessage, logType;
    DecoderInCallback callback = nullptr;
    // TODO: add read from memory option
    params.uri = videoPath;
    logType = "file";
    logMessage = videoPath;
    

    // get a decoder
    SyncDecoder decoder;
    bool succeeded;

    cout << "Video decoding to gather metadata from " << logType << " [" << logMessage
          << "] has started";
    
    std::vector<StreamMetadata> videoStreams, audioStreams;
    std::vector<DecoderMetadata> metadata;
    if ((succeeded = decoder.init(params, std::move(callback), &metadata))) {
        for (const auto& header : metadata) {
            cout << "Decoding stream of" << header.format.type ;
        
            // generate streamMetadata object
            StreamMetadata streamInfo;
            // parse stream timebase
            torch::Tensor timeBase = torch::zeros({1}, torch::kFloat);
            float * timeBaseData = timeBase.data_ptr<float>();
            timeBaseData[0] = header.num / header.den;
            streamInfo.timeBase = timeBase;
            // parse stream duration
            torch::Tensor duration = torch::zeros({1}, torch::kFloat);
            float* durationData = duration.data_ptr<float>();
            durationData[0] = (float) header.duration;
            // to get duration in seconds multiply duration by timebase
            streamInfo.duration = duration * streamInfo.timeBase;
            
            if (header.format.type == TYPE_VIDEO) {
                // parse stream fps
                torch::Tensor frameRate = torch::zeros({1}, torch::kFloat);
                float* frameRateData = frameRate.data_ptr<float>();
                frameRateData[0] = header.fps;
                streamInfo.frameRate = frameRate;
                videoStreams.push_back(streamInfo);
            } else if (header.format.type == TYPE_AUDIO) {
                const auto& format = header.format.format.audio;
                // parse stream fps
                torch::Tensor frameRate = torch::zeros({1}, torch::kFloat);
                float* frameRateData = frameRate.data_ptr<float>();
                frameRateData[0] = (float) format.samples; // this is user defined? 
                streamInfo.frameRate = frameRate;
                audioStreams.push_back(streamInfo);
            };
        }
        VideoMetadata.insert({"video", videoStreams});
        VideoMetadata.insert({"audio", audioStreams});
    } 
} //video

std::map<std::string, std::vector<StreamMetadata>> Video::getMetadata(){
    return VideoMetadata;
}


    VLOG(1) << "Video decoding from " << logType << " [" << logMessage
          << "] has started";

    DecoderMetadata audioMetadata, videoMetadata, dataMetadata;
    std::vector<DecoderMetadata> metadata;
    if ((succeeded = decoder.init(params, std::move(callback), &metadata))) {
        for (const auto& header : metadata) {
            VLOG(1) << "Decoding stream of" << header.format.type ;
        if (header.format.type == TYPE_VIDEO) {
            videoMetadata = header;
        } else if (header.format.type == TYPE_AUDIO) {
            audioMetadata = header;
        } else {
            dataMetadata = header;
        };
        }
    } 
} //video

// void Video::Seek(float time_s, std::string stream="", bool any_frame=False){
// }

// torch::List<torch::Tensor> Video::Next(){
//     return
// }



// }; // namespace video
