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
    
    
    
    std::tuple<std::string, long> current_stream; // streaam type, id
    // global video metadata
    std::map<std::string, std::vector<double>> streamFPS;
    std::map<std::string, std::vector<double>> streamDuration;
    public:
        Video(std::string videoPath, std::string stream, bool isReadFile);
        std::tuple<std::string, int64_t> getCurrentStream() const;
        std::vector<double> getDuration(std::string stream="") const;
        std::vector<double> getFPS(std::string stream="") const;
        void Seek(double ts, bool any_frame);
        torch::List<torch::Tensor> Next(std::string stream);

    private:
        bool video_any_frame=false; // add this to input parameters?
        bool succeeded=false; // decoder init flag
        // seekTS and doSeek act as a flag - if it's not set, next function simply
        // retruns the next frame. If it's set, we look at the global seek
        // time in comination with any_frame settings
        double seekTS=-1; 
        bool doSeek=false;

        void _getDecoderParams(int64_t videoStartS, int64_t getPtsOnly, std::string stream, long stream_id, bool all_streams, double seekFrameMarginUs); // this needs to be improved
        bool _setCurrentStream();
        std::map<std::string, std::vector<double>> streamTimeBase; // not used

        DecoderInCallback callback = nullptr;;
        std::vector<DecoderMetadata> metadata;
                
    protected:
        SyncDecoder decoder;
        DecoderParameters params;
    

}; // struct Video


#endif  // VIDEO_H_
