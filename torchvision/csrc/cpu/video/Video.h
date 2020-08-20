#pragma once

#ifndef VIDEO_H_
#define VIDEO_H_


#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <Python.h>
#include <c10/util/Logging.h>
#include <torch/script.h>



#include <exception>
#include "sync_decoder.h"
#include "memory_buffer.h"
#include "defs.h"

using namespace ffmpeg;


struct StreamMetadata{
    torch::Tensor frameRate;  // average frame rate for the video (float)
    torch::Tensor duration; // real world video duration in seconds (float)
    // torch::Tensor startTime; // video start time in seconds (float)
    torch::Tensor timeBase;
    // do we need a constructor here?
    explicit StreamMetadata(){
        torch::Tensor frameRate = torch::zeros({0}, torch::kFloat);
        torch::Tensor duration = torch::zeros({0}, torch::kFloat);
        torch::Tensor timeBase = torch::zeros({0}, torch::kFloat); 
    }
};



struct Video : torch::CustomClassHolder {
    // metadata is defined as a dictionary where every 
    // type has a vector containing metadata for that stream
    std::map<std::string, std::vector<StreamMetadata>> VideoMetadata;
    
    Video(std::string videoPath, std::string stream, bool isReadFile);
    std::map<std::string, std::vector<StreamMetadata>> getMetadata();
        // std::map<std::string, std::vector<StreamMetadata>> getMetadata();
        // void Seek(double ts, std::string stream="", bool any_frame=False);
        // torch::List<torch::Tensor> Next(std::string stream="")
        // torch::List<torch::Tensor> Peak(std::string stream="")
    // protected:
        // AV container type (check in decoder for exact type)
    DecoderParameters params;
        // int64_t SecToStream(double ts); // TODO: add stream type
        // float StreamToSec(int64_t pts); // TODO: add stream type
    void _getDecoderParams(int64_t videoStartUs, int64_t getPtsOnly, int stream_id, bool all_streams, double seekFrameMarginUs); // this needs to be improved
}; // struct Video


#endif  // VIDEO_H_
