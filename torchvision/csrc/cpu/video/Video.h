#pragma once

#ifndef VIDEO_H_
#define VIDEO_H_


#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <Python.h>
#include <c10/util/Logging.h>
#include <torch/script.h>


#include "../decoder/Stream.h"



struct VideoMetadata{
    double videoFps;  // average frame rate for the video (float)
    double videoDuration; // real world video duration in seconds (float)
    double videoStartTime; // video start time in seconds (float)
    // do we need a constructor here?
}

class Video {
    std::vector<VideoMetadata> Metadata;
    std::vector<Stream> AvailStreams;  // TODO: add stream type
    public:
        Video(std::string filename, std::string stream="video");
        void Seek(double ts, std::string stream="", bool any_frame=False);
        torch::List<torch::Tensor> Next(std::string stream="")
        torch::List<torch::Tensor> Peak(std::string stream="")
    protected:
        // AV container type (check in decoder for exact type)
    private:
        int64_t SecToStream(double ts); // TODO: add stream type
        float StreamToSec(int64_t pts); // TODO: add stream type
        void SetVideoStream(std::string stream="video:0")  // this needs to be improved
} // class Video

#endif  // VIDEO_H_
