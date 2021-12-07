#include <time.h>
#include "gpu_decoder.h"

GPUDecoder::GPUDecoder(std::string src_file, bool useDevFrame) : demuxer(src_file.c_str())
{
  if (cudaSuccess != cudaSetDevice(0)) {
   printf("Error setting device\n");
   return;
  }
  CheckForCudaErrors(
    cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN, 0),
    __LINE__);
  dec.init(ctx, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), useDevFrame);
  demux_time = 0.0;
  decode_time = 0.0;
  totalFrames = 0;
}

GPUDecoder::~GPUDecoder()
{
  dec.release();
  cuCtxDestroy(ctx);
}

torch::Tensor GPUDecoder::decode()
{
  uint8_t *video;
  torch::Tensor framesReturned;
  int64_t numFrames;
  clock_t start, end;
  double cpu_time_used;
  start = clock();
  demuxer.Demux(&video, &videoBytes);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  demux_time += cpu_time_used;
  framesReturned = dec.Decode(video, videoBytes);
  numFrames = dec.GetNumDecodedFrames();
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  decode_time += cpu_time_used;
  totalFrames += numFrames;
  return framesReturned;
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
    .def(torch::init<std::string, bool>())
    .def("decode", &GPUDecoder::decode)
    .def("getDecodeTime", &GPUDecoder::getDecodeTime)
    .def("getDemuxTime", &GPUDecoder::getDemuxTime)
    .def("getDemuxedBytes", &GPUDecoder::getDemuxedBytes)
    .def("getTotalFramesDecoded", &GPUDecoder::getTotalFramesDecoded)
    ;
  }
