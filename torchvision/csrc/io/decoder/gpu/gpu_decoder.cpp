#include "gpu_decoder.h"

GPUDecoder::GPUDecoder(std::string src_file, int64_t dev) : demuxer(src_file.c_str()), device(dev)
{
  if (cudaSuccess != cudaSetDevice(device)) {
   printf("Error setting device\n");
   return;
  }
  CheckForCudaErrors(
    cuDevicePrimaryCtxRetain(&ctx, device),
    __LINE__);
  dec.init(ctx, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
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
  unsigned long videoBytes = 0;
  uint8_t *frame = nullptr, *video = nullptr;
  do
  {
    demuxer.Demux(&video, &videoBytes);
    dec.Decode(video, videoBytes);
    frame = dec.FetchFrame();
  } while (frame == nullptr && videoBytes > 0);
  if (frame == nullptr) {
    auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
    return torch::zeros({0}, options);
  }
  auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
  frameTensor = torch::from_blob(
    frame, {dec.GetFrameSize()}, [](auto p) { cuMemFree((CUdeviceptr)p); }, options);
  return frameTensor;
}

torch::Tensor GPUDecoder::NV12ToYUV420(torch::Tensor frameTensor)
{
  int width = dec.GetWidth(), height = dec.GetHeight();
  int pitch = width;
  uint8_t *frame = frameTensor.data_ptr<uint8_t>();
  uint8_t *ptr = new uint8_t[((width + 1) / 2) * ((height + 1) / 2)];

  // sizes of source surface plane
  int sizePlaneY = pitch * height;
  int sizePlaneU = ((pitch + 1) / 2) * ((height + 1) / 2);
  int sizePlaneV = sizePlaneU;

  uint8_t *uv = frame + sizePlaneY;
  uint8_t *u = uv;
  uint8_t *v = uv + sizePlaneU;

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
          memcpy(v + ((pitch + 1) / 2) * i, ptr + ((width + 1) / 2) * i, ((width + 1) / 2) * sizeof(uint8_t));
      }
  }
  delete[] ptr;
  return frameTensor;
}

TORCH_LIBRARY(torchvision, m) {
  m.class_<GPUDecoder>("GPUDecoder")
    .def(torch::init<std::string, int64_t>())
    .def("next", &GPUDecoder::decode)
    .def("reformat", &GPUDecoder::NV12ToYUV420)
    ;
  }
