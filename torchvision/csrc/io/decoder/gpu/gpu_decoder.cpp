#include "gpu_decoder.h"
#include <c10/cuda/CUDAGuard.h>

/* Set cuda device, create cuda context and initialise the demuxer and decoder.
 */
GPUDecoder::GPUDecoder(std::string src_file, torch::Device dev)
    : demuxer(src_file.c_str()) {
  at::cuda::CUDAGuard device_guard(dev);
  device = device_guard.current_device().index();
  check_for_cuda_errors(
      cuDevicePrimaryCtxRetain(&ctx, device), __LINE__, __FILE__);
  decoder.init(ctx, ffmpeg_to_codec(demuxer.get_video_codec()));
  initialised = true;
}

GPUDecoder::~GPUDecoder() {
  at::cuda::CUDAGuard device_guard(device);
  decoder.release();
  if (initialised) {
    check_for_cuda_errors(
        cuDevicePrimaryCtxRelease(device), __LINE__, __FILE__);
  }
}

/* Fetch a decoded frame tensor after demuxing and decoding.
 */
torch::Tensor GPUDecoder::decode() {
  torch::Tensor frameTensor;
  unsigned long videoBytes = 0;
  uint8_t* video = nullptr;
  at::cuda::CUDAGuard device_guard(device);
  torch::Tensor frame;
  do {
    demuxer.demux(&video, &videoBytes);
    decoder.decode(video, videoBytes);
    frame = decoder.fetch_frame();
  } while (frame.numel() == 0 && videoBytes > 0);
  return frame;
}

/* Seek to a passed timestamp. The second argument controls whether to seek to a
 * keyframe.
 */
void GPUDecoder::seek(double timestamp, bool keyframes_only) {
  int flag = keyframes_only ? 0 : AVSEEK_FLAG_ANY;
  demuxer.seek(timestamp, flag);
}

c10::Dict<std::string, c10::Dict<std::string, double>> GPUDecoder::
    get_metadata() const {
  c10::Dict<std::string, c10::Dict<std::string, double>> metadata;
  c10::Dict<std::string, double> video_metadata;
  video_metadata.insert("duration", demuxer.get_duration());
  video_metadata.insert("fps", demuxer.get_fps());
  metadata.insert("video", video_metadata);
  return metadata;
}

TORCH_LIBRARY(torchvision, m) {
  m.class_<GPUDecoder>("GPUDecoder")
      .def(torch::init<std::string, torch::Device>())
      .def("seek", &GPUDecoder::seek)
      .def("get_metadata", &GPUDecoder::get_metadata)
      .def("next", &GPUDecoder::decode);
}
