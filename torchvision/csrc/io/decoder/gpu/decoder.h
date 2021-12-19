#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include <cstdint>
#include <queue>
#include <sstream>

static auto check_for_cuda_errors = [](CUresult result, int lineNum) {
  if (CUDA_SUCCESS != result) {
    std::stringstream errorStream;
    const char* errorName = nullptr;

    errorStream << __FILE__ << ":" << lineNum << std::endl;
    if (CUDA_SUCCESS != cuGetErrorName(result, &errorName)) {
      errorStream << "CUDA error with code " << result << std::endl;
    } else {
      errorStream << "CUDA error: " << errorName << std::endl;
    }
    throw std::runtime_error(errorStream.str());
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
  unsigned long decode(const uint8_t*, unsigned long);
  void release();
  uint8_t* fetch_frame();
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
  CUcontext cuContext = NULL;
  CUvideoctxlock ctxLock;
  CUvideoparser parser = NULL;
  CUvideodecoder decoder = NULL;
  CUstream cuvidStream = 0;
  int numDecodedFrames = 0;
  unsigned int numChromaPlanes = 0;
  // dimension of the output
  unsigned int width = 0, lumaHeight = 0, chromaHeight = 0;
  cudaVideoCodec videoCodec = cudaVideoCodec_NumCodecs;
  cudaVideoChromaFormat videoChromaFormat = cudaVideoChromaFormat_420;
  cudaVideoSurfaceFormat videoOutputFormat = cudaVideoSurfaceFormat_NV12;
  int bitDepthMinus8 = 0;
  int bytesPerPixel = 1;
  CUVIDEOFORMAT videoFormat = {};
  unsigned int maxWidth = 0, maxHeight = 0;
  // height of the mapped surface
  int surfaceHeight = 0;
  int surfaceWidth = 0;
  Rect displayRect = {};
  unsigned int operatingPoint = 0;
  bool dispAllLayers = false;
  int decodePicCount = 0, picNumInDecodeOrder[32];
  bool reconfigExternal = false;
  bool reconfigExtPPChange = false;
  std::queue<uint8_t*> decodedFrames;

  static int video_sequence_handler(
      void* pUserData,
      CUVIDEOFORMAT* pVideoFormat) {
    return ((Decoder*)pUserData)->handle_video_sequence(pVideoFormat);
  }
  static int picture_decode_handler(
      void* pUserData,
      CUVIDPICPARAMS* pPicParams) {
    return ((Decoder*)pUserData)->handle_picture_decode(pPicParams);
  }
  static int picture_display_handler(
      void* pUserData,
      CUVIDPARSERDISPINFO* pDispInfo) {
    return ((Decoder*)pUserData)->handle_picture_display(pDispInfo);
  }
  static int operating_point_handler(
      void* pUserData,
      CUVIDOPERATINGPOINTINFO* pOPInfo) {
    return ((Decoder*)pUserData)->get_operating_point(pOPInfo);
  }

  void query_hardware(CUVIDEOFORMAT* videoFormat);
  int reconfigure_decoder(CUVIDEOFORMAT* pVideoFormat);
  int handle_video_sequence(CUVIDEOFORMAT* pVideoFormat);
  int handle_picture_decode(CUVIDPICPARAMS* pPicParams);
  int handle_picture_display(CUVIDPARSERDISPINFO* pDispInfo);
  int get_operating_point(CUVIDOPERATINGPOINTINFO* pOPInfo);
};
