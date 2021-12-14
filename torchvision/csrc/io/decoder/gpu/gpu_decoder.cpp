#include <time.h>
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
  demux_time = 0.0;
  decode_time = 0.0;
  totalFrames = 0;
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
  uint8_t *video;
  torch::Tensor frameTensor;
  int64_t numFrames;
  clock_t start, end;
  double cpu_time_used1, cpu_time_used2;
  uint8_t *frame = nullptr;
  do
  {
    start = clock();
    demuxer.Demux(&video, &videoBytes);
    end = clock();
    cpu_time_used1 = ((double) (end - start)) / CLOCKS_PER_SEC;
    start = clock();
    numFrames = dec.Decode(video, videoBytes);
    end = clock();
    frame = dec.FetchFrame();
    cpu_time_used2 = ((double) (end - start)) / CLOCKS_PER_SEC;
  } while (frame == nullptr && videoBytes > 0);
  demux_time += cpu_time_used1;
  decode_time += cpu_time_used2;
  totalFrames += numFrames;
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

int64_t GPUDecoder::getDemuxedBytes()
{
  return videoBytes;
}

double GPUDecoder::getDecodeTime()
{
  return decode_time;
}

double GPUDecoder::getDemuxTime()
{
  return demux_time;
}

int64_t GPUDecoder::getTotalFramesDecoded()
{
  return totalFrames;
}

TORCH_LIBRARY(torchvision, m) {
  m.class_<GPUDecoder>("GPUDecoder")
    .def(torch::init<std::string, bool, int64_t, std::string>())
    .def("decode", &GPUDecoder::decode)
    .def("getDecodeTime", &GPUDecoder::getDecodeTime)
    .def("getDemuxTime", &GPUDecoder::getDemuxTime)
    .def("getDemuxedBytes", &GPUDecoder::getDemuxedBytes)
    .def("getTotalFramesDecoded", &GPUDecoder::getTotalFramesDecoded)
    ;
  }
