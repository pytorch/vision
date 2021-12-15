#include <torch/torch.h>
#include "gpu_decoder.h"

GPUDecoder::GPUDecoder(std::string src_file, bool useDevFrame, int64_t dev, std::string out_format) : demuxer(src_file.c_str()), device(dev), output_format(out_format)
{
  if (cudaSuccess != cudaSetDevice(device)) {
   printf("Error setting device\n");
   return;
  }
  CheckForCudaErrors(
    cuDevicePrimaryCtxRetain(&ctx, device),
    __LINE__);
  dec.init(ctx, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), useDevFrame);
  initialised = true;
}

GPUDecoder::~GPUDecoder()
{
  dec.release();
  if (initialised) {
    CheckForCudaErrors(
      cuDevicePrimaryCtxRelease(device),
      __LINE__);
  }
}

torch::Tensor GPUDecoder::decode()
{
  torch::Tensor frameTensor;
  int64_t videoBytes, numFrames;
  uint8_t *frame = nullptr, *video = nullptr;
  do
  {
    demuxer.Demux(&video, &videoBytes);
    numFrames = dec.Decode(video, videoBytes);
    frame = dec.FetchFrame();
  } while (frame == nullptr && videoBytes > 0);
  if (frame == nullptr) {
    auto options = torch::TensorOptions().dtype(torch::kU8).device(dec.UseDeviceFrame() ? torch::kCUDA : torch::kCPU);
    return torch::zeros({0}, options);
  }
  if (dec.UseDeviceFrame()) {
      auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
      frameTensor = torch::from_blob(
        frame, {dec.GetFrameSize()}, [](auto p) { cuMemFree((CUdeviceptr)p); }, options);
  } else {
      if (output_format == "yuv420") {
        NV12ToYUV420(frame, dec.GetWidth(), dec.GetHeight());
      }
      auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU);
      frameTensor = torch::from_blob(
        frame, {dec.GetFrameSize()}, [](auto p) { free(p); }, options);
  }
  return frameTensor;
}

TORCH_LIBRARY(torchvision, m) {
  m.class_<GPUDecoder>("GPUDecoder")
    .def(torch::init<std::string, bool, int64_t, std::string>())
    .def("next", &GPUDecoder::decode)
    ;
  }
