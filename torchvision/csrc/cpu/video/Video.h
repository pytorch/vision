#pragma once

#ifndef VIDEO_H_
#define VIDEO_H_


#include <string>
#include <vector>
#include <regex>
#include <map>

#include <ATen/ATen.h>
#include <Python.h>
#include <c10/util/Logging.h>
#include <torch/script.h>


#include <exception>
#include "sync_decoder.h"
#include "memory_buffer.h"
#include "defs.h"


using namespace ffmpeg;



struct Video : torch::CustomClassHolder {
    bool any_frame=false; // add this to input parameters
    bool succeeded=false; // this is decoder init stuff
    std::tuple<std::string, int64_t> current_stream;
    std::map<std::string, std::vector<double>> streamFPS;
    std::map<std::string, std::vector<double>> streamDuration;
    public:
        Video(std::string videoPath, std::string stream, bool isReadFile);
        std::tuple<std::string, int64_t> getCurrentStream() const;
        std::vector<double> getDuration(std::string stream="") const;
        std::vector<double> getFPS(std::string stream="") const;
        int64_t Seek(double ts, std::string stream, bool any_frame);
        int64_t Next(std::string stream); //torch::List<torch::Tensor>

    private:
        void _getDecoderParams(int64_t videoStartUs, int64_t getPtsOnly, std::string stream, long stream_id, bool all_streams, double seekFrameMarginUs); // this needs to be improved
        std::map<std::string, std::vector<double>> streamTimeBase;

        SyncDecoder decoder;
        DecoderParameters params;

        DecoderInCallback callback = nullptr;;
        std::vector<DecoderMetadata> metadata;
    // std::map<std::string, std::vector<std::map<std::string, double>>> getMetadata() const;
        // std::map<std::string, std::vector<StreamMetadata>> getMetadata();
        
        // torch::List<torch::Tensor> Peak(std::string stream="")
    protected:
        // AV container type (check in decoder for exact type)

        // int64_t SecToStream(double ts); // TODO: add stream type
        // double StreamToSec(int64_t pts); // TODO: add stream type
    

}; // struct Video


#endif  // VIDEO_H_
