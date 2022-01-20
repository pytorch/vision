#include <torch/custom_class.h>
#include <torch/torch.h>
#include "decoder.h"
#include "demuxer.h"

class GPUDecoder : public torch::CustomClassHolder {
 public:
  GPUDecoder(std::string, int64_t);
  ~GPUDecoder();
  torch::Tensor decode();

 private:
  Demuxer demuxer;
  CUcontext ctx;
  Decoder decoder;
  int64_t device;
  bool initialised = false;
};
