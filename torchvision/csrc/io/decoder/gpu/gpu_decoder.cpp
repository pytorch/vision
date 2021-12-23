#include "gpu_decoder.h"
#include <c10/cuda/CUDAGuard.h>

/* Set cuda device, create cuda context and initialise the demuxer and decoder.
 */
GPUDecoder::GPUDecoder(std::string src_file, int64_t dev)
    : demuxer(src_file.c_str()), device(dev) {
  at::cuda::CUDAGuard device_guard(device);
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
  auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
  torch::Tensor frame = torch::zeros({0}, options);
  do {
    demuxer.demux(&video, &videoBytes);
    decoder.decode(video, videoBytes);
    frame = decoder.fetch_frame();
  } while (frame.numel() == 0 && videoBytes > 0);
  return frame;
}

/* Convert a tensor with data in NV12 format to a tensor with data in YUV420
 * format in-place.
 */
torch::Tensor GPUDecoder::nv12_to_yuv420(torch::Tensor frameTensor) {
  int width = decoder.get_width(), height = decoder.get_height();
  int pitch = width;
  uint8_t* frame = frameTensor.data_ptr<uint8_t>();
  uint8_t* ptr = new uint8_t[((width + 1) / 2) * ((height + 1) / 2)];

  // sizes of source surface plane
  int sizePlaneY = pitch * height;
  int sizePlaneU = ((pitch + 1) / 2) * ((height + 1) / 2);
  int sizePlaneV = sizePlaneU;

  uint8_t* uv = frame + sizePlaneY;
  uint8_t* u = uv;
  uint8_t* v = uv + sizePlaneU;

  // split chroma from interleave to planar
  for (int y = 0; y < (height + 1) / 2; y++) {
    for (int x = 0; x < (width + 1) / 2; x++) {
      u[y * ((pitch + 1) / 2) + x] = uv[y * pitch + x * 2];
      ptr[y * ((width + 1) / 2) + x] = uv[y * pitch + x * 2 + 1];
    }
  }
  if (pitch == width) {
    memcpy(v, ptr, sizePlaneV * sizeof(uint8_t));
  } else {
    for (int i = 0; i < (height + 1) / 2; i++) {
      memcpy(
          v + ((pitch + 1) / 2) * i,
          ptr + ((width + 1) / 2) * i,
          ((width + 1) / 2) * sizeof(uint8_t));
    }
  }
  delete[] ptr;
  return frameTensor;
}

TORCH_LIBRARY(torchvision, m) {
  m.class_<GPUDecoder>("GPUDecoder")
      .def(torch::init<std::string, int64_t>())
      .def("next", &GPUDecoder::decode)
      .def("reformat", &GPUDecoder::nv12_to_yuv420);
}
