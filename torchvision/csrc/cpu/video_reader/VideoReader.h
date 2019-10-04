#pragma once

#include <torch/script.h>

// Interface for Python

/*
  return:
    videoFrame: tensor (N, H, W, C) kByte
    videoFramePts: tensor (N) kLong
    videoTimeBase: tensor (2) kInt
    videoFps: tensor (1) kFloat
    audioFrame: tensor (N, C) kFloat
    audioFramePts: tensor (N) kLong
    audioTimeBase: tensor (2) kInt
    audioSampleRate: tensor (1) kInt
*/
torch::List<torch::Tensor> readVideoFromMemory(
    // 1D tensor of data type uint8, storing the comparessed video data
    torch::Tensor input_video,
    // seeking frame in the video/audio stream is imprecise so seek to a
    // timestamp earlier by a margin The unit of margin is second
    double seekFrameMargin,
    // If only pts is needed and video/audio frames are not needed, set it
    // to 1
    int64_t getPtsOnly,
    // bool variable. Set it to 1 if video stream should be read. Otherwise, set
    // it to 0
    int64_t readVideoStream,
    /*
    Valid parameters values for rescaling video frames
    ___________________________________________________
    |  width  |  height  | min_dimension |  algorithm |
    |_________________________________________________|
    |  0  |  0  |     0        |   original           |
    |_________________________________________________|
    |  0  |  0  |     >0       |scale to min dimension|
    |_____|_____|____________________________________ |
    |  >0 |  0  |     0        |   scale keeping W    |
    |_________________________________________________|
    |  0  |  >0 |     0        |   scale keeping H    |
    |_________________________________________________|
    |  >0 |  >0 |     0        |   stretch/scale      |
    |_________________________________________________|
    */
    int64_t width,
    int64_t height,
    int64_t minDimension,
    // video frames with pts in [videoStartPts, videoEndPts] will be decoded
    // For decoding all video frames, use [0, -1]
    int64_t videoStartPts,
    int64_t videoEndPts,
    // numerator and denominator of time base of video stream.
    // For decoding all video frames, supply dummy 0 (numerator) and 1
    // (denominator). For decoding localized video frames, need to supply
    // them which will be checked during decoding
    int64_t videoTimeBaseNum,
    int64_t videoTimeBaseDen,
    // bool variable. Set it to 1 if audio stream should be read. Otherwise, set
    // it to 0
    int64_t readAudioStream,
    // audio stream sampling rate.
    // If not resampling audio waveform, supply 0
    // Otherwise, supply a positive integer.
    int64_t audioSamples,
    // audio stream channels
    // Supply 0 to use the same number of channels as in the original audio
    // stream
    int64_t audioChannels,
    // audio frames with pts in [audioStartPts, audioEndPts] will be decoded
    // For decoding all audio frames, use [0, -1]
    int64_t audioStartPts,
    int64_t audioEndPts,
    // numerator and denominator of time base of audio stream.
    // For decoding all audio frames, supply dummy 0 (numerator) and 1
    // (denominator). For decoding localized audio frames, need to supply
    // them which will be checked during decoding
    int64_t audioTimeBaseNum,
    int64_t audioTimeBaseDen);

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
    int64_t audioTimeBaseDen);
