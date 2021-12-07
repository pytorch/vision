#include <torch/custom_class.h>
#include "decoder.h"
#include "demuxer.h"

class GPUDecoder : public torch::CustomClassHolder {
  public:
    GPUDecoder(std::string, bool);
    ~GPUDecoder();
    torch::Tensor decode();
    double getDecodeTime();
    double getDemuxTime();
    int64_t getDemuxedBytes();
    int64_t getTotalFramesDecoded();

  private:
    Demuxer demuxer;
    CUcontext ctx;
    Decoder dec;
    double demux_time, decode_time;
    int64_t totalFrames, videoBytes;
};
