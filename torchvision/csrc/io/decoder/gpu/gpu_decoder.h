#include <torch/custom_class.h>
#include <torch/torch.h>
#include "decoder.h"
#include "demuxer.h"

class GPUDecoder : public torch::CustomClassHolder {
 public:
  GPUDecoder(std::string, torch::Device);
  ~GPUDecoder();
  torch::Tensor decode();
  void seek(double, bool);
  c10::Dict<std::string, c10::Dict<std::string, double>> get_metadata() const;

 private:
  Demuxer demuxer;
  CUcontext ctx;
  Decoder decoder;
  int64_t device;
  bool initialised = false;
};
