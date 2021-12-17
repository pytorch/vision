#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include <cstdint>
#include <queue>
#include <sstream>

static auto CheckForCudaErrors = [](CUresult result, int lineNum) {
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
  unsigned long Decode(const uint8_t*, unsigned long);
  void release();
  uint8_t* FetchFrame();
  cudaVideoSurfaceFormat GetOutputFormat() const {
    return videoOutputFormat;
  }
  int GetFrameSize() const {
    return GetWidth() * (lumaHeight + (chromaHeight * numChromaPlanes)) *
        bytesPerPixel;
  }
  int GetWidth() const {
    return (videoOutputFormat == cudaVideoSurfaceFormat_NV12 ||
            videoOutputFormat == cudaVideoSurfaceFormat_P016)
        ? (width + 1) & ~1
        : width;
  }
  int GetHeight() const {
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

  static int CUDAAPI
  HandleVideoSequenceProc(void* pUserData, CUVIDEOFORMAT* pVideoFormat) {
    return ((Decoder*)pUserData)->HandleVideoSequence(pVideoFormat);
  }
  static int CUDAAPI
  HandlePictureDecodeProc(void* pUserData, CUVIDPICPARAMS* pPicParams) {
    return ((Decoder*)pUserData)->HandlePictureDecode(pPicParams);
  }
  static int CUDAAPI
  HandlePictureDisplayProc(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
    return ((Decoder*)pUserData)->HandlePictureDisplay(pDispInfo);
  }
  static int CUDAAPI
  HandleOperatingPointProc(void* pUserData, CUVIDOPERATINGPOINTINFO* pOPInfo) {
    return ((Decoder*)pUserData)->GetOperatingPoint(pOPInfo);
  }

  void queryHardware(CUVIDEOFORMAT* videoFormat);
  int ReconfigureDecoder(CUVIDEOFORMAT* pVideoFormat);
  int HandleVideoSequence(CUVIDEOFORMAT* pVideoFormat);
  int HandlePictureDecode(CUVIDPICPARAMS* pPicParams);
  int HandlePictureDisplay(CUVIDPARSERDISPINFO* pDispInfo);
  int GetOperatingPoint(CUVIDOPERATINGPOINTINFO* pOPInfo);
};
