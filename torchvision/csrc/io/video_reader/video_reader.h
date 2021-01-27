#pragma once

#include <torch/types.h>

namespace vision {
namespace video_reader {

torch::List<torch::Tensor> read_video_from_memory(
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
    int64_t audioTimeBaseDen);

torch::List<torch::Tensor> read_video_from_file(
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
    int64_t audioTimeBaseDen);

torch::List<torch::Tensor> probe_video_from_memory(torch::Tensor input_video);

torch::List<torch::Tensor> probe_video_from_file(std::string videoPath);

} // namespace video_reader
} // namespace vision
