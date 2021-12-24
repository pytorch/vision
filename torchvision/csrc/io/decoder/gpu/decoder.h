#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include <torch/torch.h>
#include <cstdint>
#include <queue>

static auto check_for_cuda_errors =
    [](CUresult result, int line_num, std::string file_name) {
      if (CUDA_SUCCESS != result) {
        const char* error_name = nullptr;

        TORCH_CHECK(
            CUDA_SUCCESS != cuGetErrorName(result, &error_name),
            "CUDA error: ",
            error_name,
            " in ",
            file_name,
            " at line ",
            line_num)
        TORCH_CHECK(
            false, "Error: ", result, " in ", file_name, " at line ", line_num);
      }
    };

struct Rect {
  int left, top, right, bottom;
};

class Decoder {
 public:
  Decoder() {}
  ~Decoder();
  void init(CUcontext, cudaVideoCodec);
  void release();
  void decode(const uint8_t*, unsigned long);
  torch::Tensor fetch_frame();
  int get_frame_size() const {
    return get_width() * (lumaHeight + (chromaHeight * numChromaPlanes)) *
        bytesPerPixel;
  }
  int get_width() const {
    return (videoOutputFormat == cudaVideoSurfaceFormat_NV12 ||
            videoOutputFormat == cudaVideoSurfaceFormat_P016)
        ? (width + 1) & ~1
        : width;
  }
  int get_height() const {
    return lumaHeight;
  }

 private:
  unsigned int width = 0, lumaHeight = 0, chromaHeight = 0;
  unsigned int surfaceHeight = 0, surfaceWidth = 0;
  unsigned int maxWidth = 0, maxHeight = 0;
  unsigned int numChromaPlanes = 0;
  int bitDepthMinus8 = 0, bytesPerPixel = 1;
  int decodePicCount = 0, picNumInDecodeOrder[32];
  std::queue<torch::Tensor> decoded_frames;
  CUcontext cuContext = NULL;
  CUvideoctxlock ctxLock;
  CUvideoparser parser = NULL;
  CUvideodecoder decoder = NULL;
  CUstream cuvidStream = 0;
  cudaVideoCodec videoCodec = cudaVideoCodec_NumCodecs;
  cudaVideoChromaFormat videoChromaFormat = cudaVideoChromaFormat_420;
  cudaVideoSurfaceFormat videoOutputFormat = cudaVideoSurfaceFormat_NV12;
  CUVIDEOFORMAT videoFormat = {};
  Rect displayRect = {};

  static int video_sequence_handler(
      void* user_data,
      CUVIDEOFORMAT* video_format) {
    return ((Decoder*)user_data)->handle_video_sequence(video_format);
  }
  static int picture_decode_handler(
      void* user_data,
      CUVIDPICPARAMS* pic_params) {
    return ((Decoder*)user_data)->handle_picture_decode(pic_params);
  }
  static int picture_display_handler(
      void* user_data,
      CUVIDPARSERDISPINFO* disp_info) {
    return ((Decoder*)user_data)->handle_picture_display(disp_info);
  }
  static int operating_point_handler(
      void* user_data,
      CUVIDOPERATINGPOINTINFO* operating_info) {
    return ((Decoder*)user_data)->get_operating_point(operating_info);
  }

  void query_hardware(CUVIDEOFORMAT*);
  int reconfigure_decoder(CUVIDEOFORMAT*);
  int handle_video_sequence(CUVIDEOFORMAT*);
  int handle_picture_decode(CUVIDPICPARAMS*);
  int handle_picture_display(CUVIDPARSERDISPINFO*);
  int get_operating_point(CUVIDOPERATINGPOINTINFO*);
};
