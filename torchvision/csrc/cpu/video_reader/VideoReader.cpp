#include "VideoReader.h"
#include <ATen/ATen.h>
#include <Python.h>
#include <c10/util/Logging.h>
#include <exception>
#include "FfmpegDecoder.h"
#include "FfmpegHeaders.h"
#include "util.h"

using namespace std;

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

namespace video_reader {

class UnknownPixelFormatException : public exception {
  const char* what() const throw() override {
    return "Unknown pixel format";
  }
};

int getChannels(AVPixelFormat format) {
  int numChannels = 0;
  switch (format) {
    case AV_PIX_FMT_BGR24:
    case AV_PIX_FMT_RGB24:
      numChannels = 3;
      break;
    default:
      LOG(ERROR) << "Unknown format: " << format;
      throw UnknownPixelFormatException();
  }
  return numChannels;
}

void fillVideoTensor(
    std::vector<unique_ptr<DecodedFrame>>& frames,
    torch::Tensor& videoFrame,
    torch::Tensor& videoFramePts) {
  int frameSize = 0;
  if (videoFrame.numel() > 0) {
    frameSize = videoFrame.numel() / frames.size();
  }

  int frameCount = 0;

  uint8_t* videoFrameData =
      videoFrame.numel() > 0 ? videoFrame.data_ptr<uint8_t>() : nullptr;
  int64_t* videoFramePtsData = videoFramePts.data_ptr<int64_t>();

  for (size_t i = 0; i < frames.size(); ++i) {
    const auto& frame = frames[i];
    if (videoFrameData) {
      memcpy(
          videoFrameData + (size_t)(frameCount++) * (size_t)frameSize,
          frame->frame_.get(),
          frameSize * sizeof(uint8_t));
    }
    videoFramePtsData[i] = frame->pts_;
  }
}

void getVideoMeta(
    DecoderOutput& decoderOutput,
    int& numFrames,
    int& height,
    int& width,
    int& numChannels) {
  auto& videoFrames = decoderOutput.media_data_[TYPE_VIDEO].frames_;
  numFrames = videoFrames.size();

  FormatUnion& videoFormat = decoderOutput.media_data_[TYPE_VIDEO].format_;
  height = videoFormat.video.height;
  width = videoFormat.video.width;
  numChannels = getChannels(videoFormat.video.format);
}

void fillAudioTensor(
    std::vector<unique_ptr<DecodedFrame>>& frames,
    torch::Tensor& audioFrame,
    torch::Tensor& audioFramePts) {
  if (frames.size() == 0) {
    return;
  }

  float* audioFrameData =
      audioFrame.numel() > 0 ? audioFrame.data_ptr<float>() : nullptr;
  CHECK_EQ(audioFramePts.size(0), frames.size());
  int64_t* audioFramePtsData = audioFramePts.data_ptr<int64_t>();

  int bytesPerSample = av_get_bytes_per_sample(defaultAudioSampleFormat);

  int64_t frameDataOffset = 0;
  for (size_t i = 0; i < frames.size(); ++i) {
    audioFramePtsData[i] = frames[i]->pts_;
    if (audioFrameData) {
      memcpy(
          audioFrameData + frameDataOffset,
          frames[i]->frame_.get(),
          frames[i]->frameSize_);
      frameDataOffset += (frames[i]->frameSize_ / bytesPerSample);
    }
  }
}

void getAudioMeta(
    DecoderOutput& decoderOutput,
    int64_t& numSamples,
    int64_t& channels,
    int64_t& numFrames) {
  FormatUnion& audioFormat = decoderOutput.media_data_[TYPE_AUDIO].format_;

  channels = audioFormat.audio.channels;
  CHECK_EQ(audioFormat.audio.format, AV_SAMPLE_FMT_FLT);
  int bytesPerSample = av_get_bytes_per_sample(
      static_cast<AVSampleFormat>(audioFormat.audio.format));

  // auto& audioFrames = decoderOutput.media_frames_[TYPE_AUDIO];
  auto& audioFrames = decoderOutput.media_data_[TYPE_AUDIO].frames_;
  numFrames = audioFrames.size();
  int64_t frameSizeTotal = 0;
  for (auto const& decodedFrame : audioFrames) {
    frameSizeTotal += static_cast<int64_t>(decodedFrame->frameSize_);
  }
  VLOG(2) << "numFrames: " << numFrames;
  VLOG(2) << "frameSizeTotal: " << frameSizeTotal;
  VLOG(2) << "channels: " << channels;
  VLOG(2) << "bytesPerSample: " << bytesPerSample;
  CHECK_EQ(frameSizeTotal % (channels * bytesPerSample), 0);
  numSamples = frameSizeTotal / (channels * bytesPerSample);
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
  unique_ptr<DecoderParameters> params = util::getDecoderParams(
      seekFrameMargin,
      getPtsOnly,
      readVideoStream,
      width,
      height,
      minDimension,
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

  FfmpegDecoder decoder;
  DecoderOutput decoderOutput;

  if (isReadFile) {
    decoder.decodeFile(std::move(params), videoPath, decoderOutput);
  } else {
    decoder.decodeMemory(
        std::move(params),
        input_video.data_ptr<uint8_t>(),
        input_video.size(0),
        decoderOutput);
  }

  // video section
  torch::Tensor videoFrame = torch::zeros({0}, torch::kByte);
  torch::Tensor videoFramePts = torch::zeros({0}, torch::kLong);
  torch::Tensor videoTimeBase = torch::zeros({0}, torch::kInt);
  torch::Tensor videoFps = torch::zeros({0}, torch::kFloat);
  torch::Tensor videoDuration = torch::zeros({0}, torch::kLong);

  if (readVideoStream == 1) {
    auto it = decoderOutput.media_data_.find(TYPE_VIDEO);
    if (it != decoderOutput.media_data_.end()) {
      int numVideoFrames, outHeight, outWidth, numChannels;
      getVideoMeta(
          decoderOutput, numVideoFrames, outHeight, outWidth, numChannels);

      if (getPtsOnly == 0) {
        videoFrame = torch::zeros(
            {numVideoFrames, outHeight, outWidth, numChannels}, torch::kByte);
      }

      videoFramePts = torch::zeros({numVideoFrames}, torch::kLong);

      fillVideoTensor(
          decoderOutput.media_data_[TYPE_VIDEO].frames_,
          videoFrame,
          videoFramePts);

      videoTimeBase = torch::zeros({2}, torch::kInt);
      int* videoTimeBaseData = videoTimeBase.data_ptr<int>();
      videoTimeBaseData[0] = it->second.format_.video.timeBaseNum;
      videoTimeBaseData[1] = it->second.format_.video.timeBaseDen;

      videoFps = torch::zeros({1}, torch::kFloat);
      float* videoFpsData = videoFps.data_ptr<float>();
      videoFpsData[0] = it->second.format_.video.fps;

      videoDuration = torch::zeros({1}, torch::kLong);
      int64_t* videoDurationData = videoDuration.data_ptr<int64_t>();
      videoDurationData[0] = it->second.format_.video.duration;
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
  if (readAudioStream == 1) {
    auto it = decoderOutput.media_data_.find(TYPE_AUDIO);
    if (it != decoderOutput.media_data_.end()) {
      VLOG(1) << "Find audio stream";
      int64_t numAudioSamples = 0, outAudioChannels = 0, numAudioFrames = 0;
      getAudioMeta(
          decoderOutput, numAudioSamples, outAudioChannels, numAudioFrames);
      VLOG(2) << "numAudioSamples: " << numAudioSamples;
      VLOG(2) << "outAudioChannels: " << outAudioChannels;
      VLOG(2) << "numAudioFrames: " << numAudioFrames;

      if (getPtsOnly == 0) {
        audioFrame =
            torch::zeros({numAudioSamples, outAudioChannels}, torch::kFloat);
      }
      audioFramePts = torch::zeros({numAudioFrames}, torch::kLong);
      fillAudioTensor(
          decoderOutput.media_data_[TYPE_AUDIO].frames_,
          audioFrame,
          audioFramePts);

      audioTimeBase = torch::zeros({2}, torch::kInt);
      int* audioTimeBaseData = audioTimeBase.data_ptr<int>();
      audioTimeBaseData[0] = it->second.format_.audio.timeBaseNum;
      audioTimeBaseData[1] = it->second.format_.audio.timeBaseDen;

      audioSampleRate = torch::zeros({1}, torch::kInt);
      int* audioSampleRateData = audioSampleRate.data_ptr<int>();
      audioSampleRateData[0] = it->second.format_.audio.samples;

      audioDuration = torch::zeros({1}, torch::kLong);
      int64_t* audioDurationData = audioDuration.data_ptr<int64_t>();
      audioDurationData[0] = it->second.format_.audio.duration;
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
  unique_ptr<DecoderParameters> params = util::getDecoderParams(
      0, // seekFrameMargin
      0, // getPtsOnly
      1, // readVideoStream
      0, // width
      0, // height
      0, // minDimension
      0, // videoStartPts
      0, // videoEndPts
      0, // videoTimeBaseNum
      1, // videoTimeBaseDen
      1, // readAudioStream
      0, // audioSamples
      0, // audioChannels
      0, // audioStartPts
      0, // audioEndPts
      0, // audioTimeBaseNum
      1 // audioTimeBaseDen
  );

  FfmpegDecoder decoder;
  DecoderOutput decoderOutput;
  if (isReadFile) {
    decoder.probeFile(std::move(params), videoPath, decoderOutput);
  } else {
    decoder.probeMemory(
        std::move(params),
        input_video.data_ptr<uint8_t>(),
        input_video.size(0),
        decoderOutput);
  }
  // video section
  torch::Tensor videoTimeBase = torch::zeros({0}, torch::kInt);
  torch::Tensor videoFps = torch::zeros({0}, torch::kFloat);
  torch::Tensor videoDuration = torch::zeros({0}, torch::kLong);

  auto it = decoderOutput.media_data_.find(TYPE_VIDEO);
  if (it != decoderOutput.media_data_.end()) {
    VLOG(1) << "Find video stream";
    videoTimeBase = torch::zeros({2}, torch::kInt);
    int* videoTimeBaseData = videoTimeBase.data_ptr<int>();
    videoTimeBaseData[0] = it->second.format_.video.timeBaseNum;
    videoTimeBaseData[1] = it->second.format_.video.timeBaseDen;

    videoFps = torch::zeros({1}, torch::kFloat);
    float* videoFpsData = videoFps.data_ptr<float>();
    videoFpsData[0] = it->second.format_.video.fps;

    videoDuration = torch::zeros({1}, torch::kLong);
    int64_t* videoDurationData = videoDuration.data_ptr<int64_t>();
    videoDurationData[0] = it->second.format_.video.duration;
  } else {
    VLOG(1) << "Miss video stream";
  }

  // audio section
  torch::Tensor audioTimeBase = torch::zeros({0}, torch::kInt);
  torch::Tensor audioSampleRate = torch::zeros({0}, torch::kInt);
  torch::Tensor audioDuration = torch::zeros({0}, torch::kLong);

  it = decoderOutput.media_data_.find(TYPE_AUDIO);
  if (it != decoderOutput.media_data_.end()) {
    VLOG(1) << "Find audio stream";
    audioTimeBase = torch::zeros({2}, torch::kInt);
    int* audioTimeBaseData = audioTimeBase.data_ptr<int>();
    audioTimeBaseData[0] = it->second.format_.audio.timeBaseNum;
    audioTimeBaseData[1] = it->second.format_.audio.timeBaseDen;

    audioSampleRate = torch::zeros({1}, torch::kInt);
    int* audioSampleRateData = audioSampleRate.data_ptr<int>();
    audioSampleRateData[0] = it->second.format_.audio.samples;

    audioDuration = torch::zeros({1}, torch::kLong);
    int64_t* audioDurationData = audioDuration.data_ptr<int64_t>();
    audioDurationData[0] = it->second.format_.audio.duration;
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

static auto registry = torch::RegisterOperators()
                           .op("video_reader::read_video_from_memory",
                               &video_reader::readVideoFromMemory)
                           .op("video_reader::read_video_from_file",
                               &video_reader::readVideoFromFile)
                           .op("video_reader::probe_video_from_memory",
                               &video_reader::probeVideoFromMemory)
                           .op("video_reader::probe_video_from_file",
                               &video_reader::probeVideoFromFile);
