#include <torch/custom_class.h>
#include "decoder.h"
#include "demuxer.h"

class GPUDecoder : public torch::CustomClassHolder {
  public:
    GPUDecoder(std::string, bool, int64_t, std::string);
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
    int64_t totalFrames, videoBytes, device;
    bool initialised = false;
    std::string output_format;

    void NV12ToYUV420(uint8_t *frame, int width, int height)
    {
      int pitch = width;
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
    }
};
