#include "VideoReader.h"
#include <ATen/ATen.h>
#include <Python.h>
#include <c10/util/Logging.h>
#include <exception>
#include "memory_buffer.h"
#include "sync_decoder.h"

using namespace std;
using namespace ffmpeg;

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
#ifdef _WIN32
PyMODINIT_FUNC PyInit_video_reader(void) {
  // No need to do anything.
  return NULL;
}
#endif

namespace video_reader {

const AVPixelFormat defaultVideoPixelFormat = AV_PIX_FMT_RGB24;
const AVSampleFormat defaultAudioSampleFormat = AV_SAMPLE_FMT_FLT;
const AVRational timeBaseQ = AVRational{1, AV_TIME_BASE};
const size_t decoderTimeoutMs = 600000;
// A jitter can be added to the end of the range to avoid conversion/rounding
// error, small value 100us won't be enough to select the next frame, but enough
// to compensate rounding error due to the multiple conversions.
const size_t timeBaseJitterUs = 100;

DecoderParameters getDecoderParams(
    int64_t videoStartUs,
    int64_t videoEndUs,
    double seekFrameMarginUs,
    int64_t getPtsOnly,
    int64_t readVideoStream,
    int videoWidth,
    int videoHeight,
    int videoMinDimension,
    int videoMaxDimension,
    int64_t readAudioStream,
    int audioSamples,
    int audioChannels) {
  DecoderParameters params;
  params.headerOnly = getPtsOnly != 0;
  params.seekAccuracy = seekFrameMarginUs;
  params.startOffset = videoStartUs;
  params.endOffset = videoEndUs;
  params.timeoutMs = decoderTimeoutMs;
  params.preventStaleness = false;

  if (readVideoStream == 1) {
    MediaFormat videoFormat(0);
    videoFormat.type = TYPE_VIDEO;
    videoFormat.format.video.format = defaultVideoPixelFormat;
    videoFormat.format.video.width = videoWidth;
    videoFormat.format.video.height = videoHeight;
    videoFormat.format.video.minDimension = videoMinDimension;
    videoFormat.format.video.maxDimension = videoMaxDimension;
    params.formats.insert(videoFormat);
  }

  if (readAudioStream == 1) {
    MediaFormat audioFormat;
    audioFormat.type = TYPE_AUDIO;
    audioFormat.format.audio.format = defaultAudioSampleFormat;
    audioFormat.format.audio.samples = audioSamples;
    audioFormat.format.audio.channels = audioChannels;
    params.formats.insert(audioFormat);
  }

  return params;
}

// returns number of written bytes
template <typename T>
size_t fillTensor(
    std::vector<DecoderOutputMessage>& msgs,
    torch::Tensor& frame,
    torch::Tensor& framePts,
    int64_t num,
    int64_t den) {
  if (msgs.empty()) {
    return 0;
  }
  T* frameData = frame.numel() > 0 ? frame.data_ptr<T>() : nullptr;
  int64_t* framePtsData = framePts.data_ptr<int64_t>();
  CHECK_EQ(framePts.size(0), msgs.size());
  size_t avgElementsInFrame = frame.numel() / msgs.size();

  size_t offset = 0;
  for (size_t i = 0; i < msgs.size(); ++i) {
    const auto& msg = msgs[i];
    // convert pts into original time_base
    AVRational avr = AVRational{(int)num, (int)den};
    framePtsData[i] = av_rescale_q(msg.header.pts, timeBaseQ, avr);
    VLOG(2) << "PTS type: " << sizeof(T) << ", us: " << msg.header.pts
            << ", original: " << framePtsData[i];

    if (frameData) {
      auto sizeInBytes = msg.payload->length();
      memcpy(frameData + offset, msg.payload->data(), sizeInBytes);
      if (sizeof(T) == sizeof(uint8_t)) {
        // Video - move by allocated frame size
        offset += avgElementsInFrame / sizeof(T);
      } else {
        // Audio - move by number of samples
        offset += sizeInBytes / sizeof(T);
      }
    }
  }
  return offset * sizeof(T);
}

size_t fillVideoTensor(
    std::vector<DecoderOutputMessage>& msgs,
    torch::Tensor& videoFrame,
    torch::Tensor& videoFramePts,
    int64_t num,
    int64_t den) {
  return fillTensor<uint8_t>(msgs, videoFrame, videoFramePts, num, den);
}

size_t fillAudioTensor(
    std::vector<DecoderOutputMessage>& msgs,
    torch::Tensor& audioFrame,
    torch::Tensor& audioFramePts,
    int64_t num,
    int64_t den) {
  return fillTensor<float>(msgs, audioFrame, audioFramePts, num, den);
}

void offsetsToUs(
    double& seekFrameMargin,
    int64_t readVideoStream,
    int64_t videoStartPts,
    int64_t videoEndPts,
    int64_t videoTimeBaseNum,
    int64_t videoTimeBaseDen,
    int64_t readAudioStream,
    int64_t audioStartPts,
    int64_t audioEndPts,
    int64_t audioTimeBaseNum,
    int64_t audioTimeBaseDen,
    int64_t& videoStartUs,
    int64_t& videoEndUs) {
  seekFrameMargin *= AV_TIME_BASE;
  videoStartUs = 0;
  videoEndUs = -1;

  if (readVideoStream) {
    AVRational vr = AVRational{(int)videoTimeBaseNum, (int)videoTimeBaseDen};
    if (videoStartPts > 0) {
      videoStartUs = av_rescale_q(videoStartPts, vr, timeBaseQ);
    }
    if (videoEndPts > 0) {
      // Add jitter to the end of the range to avoid conversion/rounding error.
      // Small value 100us won't be enough to select the next frame, but enough
      // to compensate rounding error due to the multiple conversions.
      videoEndUs = timeBaseJitterUs + av_rescale_q(videoEndPts, vr, timeBaseQ);
    }
  } else if (readAudioStream) {
    AVRational ar = AVRational{(int)audioTimeBaseNum, (int)audioTimeBaseDen};
    if (audioStartPts > 0) {
      videoStartUs = av_rescale_q(audioStartPts, ar, timeBaseQ);
    }
    if (audioEndPts > 0) {
      // Add jitter to the end of the range to avoid conversion/rounding error.
      // Small value 100us won't be enough to select the next frame, but enough
      // to compensate rounding error due to the multiple conversions.
      videoEndUs = timeBaseJitterUs + av_rescale_q(audioEndPts, ar, timeBaseQ);
    }
  }
}

torch::List<torch::Tensor> readVideo(
    bool isReadFile,
    const torch::Tensor& input_video,
    std::string videoPath,
    double seekFrameMargin,
    int64_t getPtsOnly,
    int64_t readVideoStream,
    int64_t width,
    int64_t height,
    int64_t minDimension,
    int64_t maxDimension,
    int64_t videoStartPts,
    int64_t videoEndPts,
    int64_t videoTimeBaseNum,
    int64_t videoTimeBaseDen,
    int64_t readAudioStream,
    int64_t audioSamples,
    int64_t audioChannels,
    int64_t audioStartPts,
    int64_t audioEndPts,
    int64_t audioTimeBaseNum,
    int64_t audioTimeBaseDen) {
  int64_t videoStartUs, videoEndUs;

  offsetsToUs(
      seekFrameMargin,
      readVideoStream,
      videoStartPts,
      videoEndPts,
      videoTimeBaseNum,
      videoTimeBaseDen,
      readAudioStream,
      audioStartPts,
      audioEndPts,
      audioTimeBaseNum,
      audioTimeBaseDen,
      videoStartUs,
      videoEndUs);

  DecoderParameters params = getDecoderParams(
      videoStartUs, // videoStartPts
      videoEndUs, // videoEndPts
      seekFrameMargin, // seekFrameMargin
      getPtsOnly, // getPtsOnly
      readVideoStream, // readVideoStream
      width, // width
      height, // height
      minDimension, // minDimension
      maxDimension, // maxDimension
      readAudioStream, // readAudioStream
      audioSamples, // audioSamples
      audioChannels // audioChannels
  );

  SyncDecoder decoder;
  std::vector<DecoderOutputMessage> audioMessages, videoMessages;
  DecoderInCallback callback = nullptr;
  std::string logMessage, logType;
  if (isReadFile) {
    params.uri = videoPath;
    logType = "file";
    logMessage = videoPath;
  } else {
    callback = MemoryBuffer::getCallback(
        input_video.data_ptr<uint8_t>(), input_video.size(0));
    logType = "memory";
    logMessage = std::to_string(input_video.size(0));
  }

  VLOG(1) << "Video decoding from " << logType << " [" << logMessage
          << "] has started";

  const auto now = std::chrono::system_clock::now();

  bool succeeded;
  DecoderMetadata audioMetadata, videoMetadata;
  std::vector<DecoderMetadata> metadata;
  if ((succeeded = decoder.init(params, std::move(callback), &metadata))) {
    for (const auto& header : metadata) {
      if (header.format.type == TYPE_VIDEO) {
        videoMetadata = header;
      } else if (header.format.type == TYPE_AUDIO) {
        audioMetadata = header;
      }
    }
    int res;
    DecoderOutputMessage msg;
    while (0 == (res = decoder.decode(&msg, decoderTimeoutMs))) {
      if (msg.header.format.type == TYPE_VIDEO) {
        videoMessages.push_back(std::move(msg));
      }
      if (msg.header.format.type == TYPE_AUDIO) {
        audioMessages.push_back(std::move(msg));
      }
      msg.payload.reset();
    }
  } else {
    LOG(ERROR) << "Decoder initialization has failed";
  }
  const auto then = std::chrono::system_clock::now();
  VLOG(1) << "Video decoding from " << logType << " [" << logMessage
          << "] has finished, "
          << std::chrono::duration_cast<std::chrono::microseconds>(then - now)
                 .count()
          << " us";

  decoder.shutdown();

  // video section
  torch::Tensor videoFrame = torch::zeros({0}, torch::kByte);
  torch::Tensor videoFramePts = torch::zeros({0}, torch::kLong);
  torch::Tensor videoTimeBase = torch::zeros({0}, torch::kInt);
  torch::Tensor videoFps = torch::zeros({0}, torch::kFloat);
  torch::Tensor videoDuration = torch::zeros({0}, torch::kLong);

  if (succeeded && readVideoStream == 1) {
    if (!videoMessages.empty()) {
      const auto& header = videoMetadata;
      const auto& format = header.format.format.video;
      int numVideoFrames = videoMessages.size();
      int outHeight = format.height;
      int outWidth = format.width;
      int numChannels = 3; // decoder guarantees the default AV_PIX_FMT_RGB24

      size_t expectedWrittenBytes = 0;
      if (getPtsOnly == 0) {
        videoFrame = torch::zeros(
            {numVideoFrames, outHeight, outWidth, numChannels}, torch::kByte);
        expectedWrittenBytes =
            (size_t)numVideoFrames * outHeight * outWidth * numChannels;
      }

      videoFramePts = torch::zeros({numVideoFrames}, torch::kLong);

      VLOG(2) << "video duration: " << header.duration
              << ", fps: " << header.fps << ", num: " << header.num
              << ", den: " << header.den << ", num frames: " << numVideoFrames;

      auto numberWrittenBytes = fillVideoTensor(
          videoMessages, videoFrame, videoFramePts, header.num, header.den);

      CHECK_EQ(numberWrittenBytes, expectedWrittenBytes);

      videoTimeBase = torch::zeros({2}, torch::kInt);
      int* videoTimeBaseData = videoTimeBase.data_ptr<int>();
      videoTimeBaseData[0] = header.num;
      videoTimeBaseData[1] = header.den;

      videoFps = torch::zeros({1}, torch::kFloat);
      float* videoFpsData = videoFps.data_ptr<float>();
      videoFpsData[0] = header.fps;

      videoDuration = torch::zeros({1}, torch::kLong);
      int64_t* videoDurationData = videoDuration.data_ptr<int64_t>();
      AVRational vr = AVRational{(int)header.num, (int)header.den};
      videoDurationData[0] = av_rescale_q(header.duration, timeBaseQ, vr);
      VLOG(1) << "Video decoding from " << logType << " [" << logMessage
              << "] filled video tensors";
    } else {
      VLOG(1) << "Miss video stream";
    }
  }

  // audio section
  torch::Tensor audioFrame = torch::zeros({0}, torch::kFloat);
  torch::Tensor audioFramePts = torch::zeros({0}, torch::kLong);
  torch::Tensor audioTimeBase = torch::zeros({0}, torch::kInt);
  torch::Tensor audioSampleRate = torch::zeros({0}, torch::kInt);
  torch::Tensor audioDuration = torch::zeros({0}, torch::kLong);
  if (succeeded && readAudioStream == 1) {
    if (!audioMessages.empty()) {
      const auto& header = audioMetadata;
      const auto& format = header.format.format.audio;

      int64_t outAudioChannels = format.channels;
      int bytesPerSample =
          av_get_bytes_per_sample(static_cast<AVSampleFormat>(format.format));

      int numAudioFrames = audioMessages.size();
      int64_t numAudioSamples = 0;
      if (getPtsOnly == 0) {
        int64_t frameSizeTotal = 0;
        for (auto const& audioMessage : audioMessages) {
          frameSizeTotal += audioMessage.payload->length();
        }

        CHECK_EQ(frameSizeTotal % (outAudioChannels * bytesPerSample), 0);
        numAudioSamples = frameSizeTotal / (outAudioChannels * bytesPerSample);

        audioFrame =
            torch::zeros({numAudioSamples, outAudioChannels}, torch::kFloat);
      }
      audioFramePts = torch::zeros({numAudioFrames}, torch::kLong);

      VLOG(2) << "audio duration: " << header.duration
              << ", channels: " << format.channels
              << ", sample rate: " << format.samples << ", num: " << header.num
              << ", den: " << header.den;

      auto numberWrittenBytes = fillAudioTensor(
          audioMessages, audioFrame, audioFramePts, header.num, header.den);
      CHECK_EQ(
          numberWrittenBytes,
          numAudioSamples * outAudioChannels * sizeof(float));

      audioTimeBase = torch::zeros({2}, torch::kInt);
      int* audioTimeBaseData = audioTimeBase.data_ptr<int>();
      audioTimeBaseData[0] = header.num;
      audioTimeBaseData[1] = header.den;

      audioSampleRate = torch::zeros({1}, torch::kInt);
      int* audioSampleRateData = audioSampleRate.data_ptr<int>();
      audioSampleRateData[0] = format.samples;

      audioDuration = torch::zeros({1}, torch::kLong);
      int64_t* audioDurationData = audioDuration.data_ptr<int64_t>();
      AVRational ar = AVRational{(int)header.num, (int)header.den};
      audioDurationData[0] = av_rescale_q(header.duration, timeBaseQ, ar);
      VLOG(1) << "Video decoding from " << logType << " [" << logMessage
              << "] filled audio tensors";
    } else {
      VLOG(1) << "Miss audio stream";
    }
  }

  torch::List<torch::Tensor> result;
  result.push_back(std::move(videoFrame));
  result.push_back(std::move(videoFramePts));
  result.push_back(std::move(videoTimeBase));
  result.push_back(std::move(videoFps));
  result.push_back(std::move(videoDuration));
  result.push_back(std::move(audioFrame));
  result.push_back(std::move(audioFramePts));
  result.push_back(std::move(audioTimeBase));
  result.push_back(std::move(audioSampleRate));
  result.push_back(std::move(audioDuration));

  VLOG(1) << "Video decoding from " << logType << " [" << logMessage
          << "] about to return";

  return result;
}

torch::List<torch::Tensor> readVideoFromMemory(
    torch::Tensor input_video,
    double seekFrameMargin,
    int64_t getPtsOnly,
    int64_t readVideoStream,
    int64_t width,
    int64_t height,
    int64_t minDimension,
    int64_t maxDimension,
    int64_t videoStartPts,
    int64_t videoEndPts,
    int64_t videoTimeBaseNum,
    int64_t videoTimeBaseDen,
    int64_t readAudioStream,
    int64_t audioSamples,
    int64_t audioChannels,
    int64_t audioStartPts,
    int64_t audioEndPts,
    int64_t audioTimeBaseNum,
    int64_t audioTimeBaseDen) {
  return readVideo(
      false,
      input_video,
      "", // videoPath
      seekFrameMargin,
      getPtsOnly,
      readVideoStream,
      width,
      height,
      minDimension,
      maxDimension,
      videoStartPts,
      videoEndPts,
      videoTimeBaseNum,
      videoTimeBaseDen,
      readAudioStream,
      audioSamples,
      audioChannels,
      audioStartPts,
      audioEndPts,
      audioTimeBaseNum,
      audioTimeBaseDen);
}

torch::List<torch::Tensor> readVideoFromFile(
    std::string videoPath,
    double seekFrameMargin,
    int64_t getPtsOnly,
    int64_t readVideoStream,
    int64_t width,
    int64_t height,
    int64_t minDimension,
    int64_t maxDimension,
    int64_t videoStartPts,
    int64_t videoEndPts,
    int64_t videoTimeBaseNum,
    int64_t videoTimeBaseDen,
    int64_t readAudioStream,
    int64_t audioSamples,
    int64_t audioChannels,
    int64_t audioStartPts,
    int64_t audioEndPts,
    int64_t audioTimeBaseNum,
    int64_t audioTimeBaseDen) {
  torch::Tensor dummy_input_video = torch::ones({0});
  return readVideo(
      true,
      dummy_input_video,
      videoPath,
      seekFrameMargin,
      getPtsOnly,
      readVideoStream,
      width,
      height,
      minDimension,
      maxDimension,
      videoStartPts,
      videoEndPts,
      videoTimeBaseNum,
      videoTimeBaseDen,
      readAudioStream,
      audioSamples,
      audioChannels,
      audioStartPts,
      audioEndPts,
      audioTimeBaseNum,
      audioTimeBaseDen);
}

torch::List<torch::Tensor> probeVideo(
    bool isReadFile,
    const torch::Tensor& input_video,
    std::string videoPath) {
  DecoderParameters params = getDecoderParams(
      0, // videoStartUs
      -1, // videoEndUs
      0, // seekFrameMargin
      1, // getPtsOnly
      1, // readVideoStream
      0, // width
      0, // height
      0, // minDimension
      0, // maxDimension
      1, // readAudioStream
      0, // audioSamples
      0 // audioChannels
  );

  SyncDecoder decoder;
  DecoderInCallback callback = nullptr;
  std::string logMessage, logType;
  if (isReadFile) {
    params.uri = videoPath;
    logType = "file";
    logMessage = videoPath;
  } else {
    callback = MemoryBuffer::getCallback(
        input_video.data_ptr<uint8_t>(), input_video.size(0));
    logType = "memory";
    logMessage = std::to_string(input_video.size(0));
  }

  VLOG(1) << "Video probing from " << logType << " [" << logMessage
          << "] has started";

  const auto now = std::chrono::system_clock::now();

  bool succeeded;
  bool gotAudio = false, gotVideo = false;
  DecoderMetadata audioMetadata, videoMetadata;
  std::vector<DecoderMetadata> metadata;
  if ((succeeded = decoder.init(params, std::move(callback), &metadata))) {
    for (const auto& header : metadata) {
      if (header.format.type == TYPE_VIDEO) {
        gotVideo = true;
        videoMetadata = header;
      } else if (header.format.type == TYPE_AUDIO) {
        gotAudio = true;
        audioMetadata = header;
      }
    }
    const auto then = std::chrono::system_clock::now();
    VLOG(1) << "Video probing from " << logType << " [" << logMessage
            << "] has finished, "
            << std::chrono::duration_cast<std::chrono::microseconds>(then - now)
                   .count()
            << " us";
  } else {
    LOG(ERROR) << "Decoder initialization has failed";
  }

  decoder.shutdown();

  // video section
  torch::Tensor videoTimeBase = torch::zeros({0}, torch::kInt);
  torch::Tensor videoFps = torch::zeros({0}, torch::kFloat);
  torch::Tensor videoDuration = torch::zeros({0}, torch::kLong);

  if (succeeded && gotVideo) {
    videoTimeBase = torch::zeros({2}, torch::kInt);
    int* videoTimeBaseData = videoTimeBase.data_ptr<int>();
    const auto& header = videoMetadata;
    const auto& media = header.format;

    videoTimeBaseData[0] = header.num;
    videoTimeBaseData[1] = header.den;

    videoFps = torch::zeros({1}, torch::kFloat);
    float* videoFpsData = videoFps.data_ptr<float>();
    videoFpsData[0] = header.fps;

    videoDuration = torch::zeros({1}, torch::kLong);
    int64_t* videoDurationData = videoDuration.data_ptr<int64_t>();
    AVRational avr = AVRational{(int)header.num, (int)header.den};
    videoDurationData[0] = av_rescale_q(header.duration, timeBaseQ, avr);

    VLOG(2) << "Prob fps: " << header.fps << ", duration: " << header.duration
            << ", num: " << header.num << ", den: " << header.den;

    VLOG(1) << "Video probing from " << logType << " [" << logMessage
            << "] filled video tensors";
  } else {
    LOG(ERROR) << "Miss video stream";
  }

  // audio section
  torch::Tensor audioTimeBase = torch::zeros({0}, torch::kInt);
  torch::Tensor audioSampleRate = torch::zeros({0}, torch::kInt);
  torch::Tensor audioDuration = torch::zeros({0}, torch::kLong);

  if (succeeded && gotAudio) {
    audioTimeBase = torch::zeros({2}, torch::kInt);
    int* audioTimeBaseData = audioTimeBase.data_ptr<int>();
    const auto& header = audioMetadata;
    const auto& media = header.format;
    const auto& format = media.format.audio;

    audioTimeBaseData[0] = header.num;
    audioTimeBaseData[1] = header.den;

    audioSampleRate = torch::zeros({1}, torch::kInt);
    int* audioSampleRateData = audioSampleRate.data_ptr<int>();
    audioSampleRateData[0] = format.samples;

    audioDuration = torch::zeros({1}, torch::kLong);
    int64_t* audioDurationData = audioDuration.data_ptr<int64_t>();
    AVRational avr = AVRational{(int)header.num, (int)header.den};
    audioDurationData[0] = av_rescale_q(header.duration, timeBaseQ, avr);

    VLOG(2) << "Prob sample rate: " << format.samples
            << ", duration: " << header.duration << ", num: " << header.num
            << ", den: " << header.den;

    VLOG(1) << "Video probing from " << logType << " [" << logMessage
            << "] filled audio tensors";
  } else {
    VLOG(1) << "Miss audio stream";
  }

  torch::List<torch::Tensor> result;
  result.push_back(std::move(videoTimeBase));
  result.push_back(std::move(videoFps));
  result.push_back(std::move(videoDuration));
  result.push_back(std::move(audioTimeBase));
  result.push_back(std::move(audioSampleRate));
  result.push_back(std::move(audioDuration));

  VLOG(1) << "Video probing from " << logType << " [" << logMessage
          << "] is about to return";

  return result;
}

torch::List<torch::Tensor> probeVideoFromMemory(torch::Tensor input_video) {
  return probeVideo(false, input_video, "");
}

torch::List<torch::Tensor> probeVideoFromFile(std::string videoPath) {
  torch::Tensor dummy_input_video = torch::ones({0});
  return probeVideo(true, dummy_input_video, videoPath);
}

} // namespace video_reader

TORCH_LIBRARY_FRAGMENT(video_reader, m) {
  m.def("read_video_from_memory", video_reader::readVideoFromMemory);
  m.def("read_video_from_file", video_reader::readVideoFromFile);
  m.def("probe_video_from_memory", video_reader::probeVideoFromMemory);
  m.def("probe_video_from_file", video_reader::probeVideoFromFile);
}
