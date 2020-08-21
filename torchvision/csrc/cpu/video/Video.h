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
    // metadata is defined as a dictionary where every 
    // type value is a list of lists that contains tuple <char: "info", double: "value">
    std::tuple<std::string, int64_t> current_stream;
    std::map<std::string, std::vector<double>> streamMetadata;
    public:
        Video(std::string videoPath, std::string stream, bool isReadFile);
        std::tuple<std::string, int64_t> getCurrentStream() const;
        std::vector<double> getFPS(std::string stream) const;

    private:
        std::tuple<std::string, int64_t> _parseStream(const std::string& streamString);
        void _getDecoderParams(int64_t videoStartUs, int64_t getPtsOnly, std::string stream, long stream_id, bool all_streams, double seekFrameMarginUs); // this needs to be improved

    // std::map<std::string, std::vector<std::map<std::string, double>>> getMetadata() const;
        // std::map<std::string, std::vector<StreamMetadata>> getMetadata();
        // void Seek(double ts, std::string stream="", bool any_frame=False);
        // torch::List<torch::Tensor> Next(std::string stream="")
        // torch::List<torch::Tensor> Peak(std::string stream="")
    protected:
        // AV container type (check in decoder for exact type)
        SyncDecoder decoder;
        DecoderParameters params;

        // int64_t SecToStream(double ts); // TODO: add stream type
        // double StreamToSec(int64_t pts); // TODO: add stream type
    
}; // struct Video


#endif  // VIDEO_H_
