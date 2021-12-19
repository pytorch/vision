#include "decoder.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <unordered_map>

static float chroma_height_factor(cudaVideoSurfaceFormat surfaceFormat) {
  return (surfaceFormat == cudaVideoSurfaceFormat_YUV444 ||
          surfaceFormat == cudaVideoSurfaceFormat_YUV444_16Bit)
      ? 1.0
      : 0.5;
}

static int chroma_plane_count(cudaVideoSurfaceFormat surfaceFormat) {
  return (surfaceFormat == cudaVideoSurfaceFormat_YUV444 ||
          surfaceFormat == cudaVideoSurfaceFormat_YUV444_16Bit)
      ? 2
      : 1;
}

void Decoder::init(CUcontext context, cudaVideoCodec codec) {
  cuContext = context;
  videoCodec = codec;
  check_for_cuda_errors(cuvidCtxLockCreate(&ctxLock, cuContext), __LINE__);

  CUVIDPARSERPARAMS parserParams = {
      .CodecType = codec,
      .ulMaxNumDecodeSurfaces = 1,
      .ulClockRate = 1000,
      .ulMaxDisplayDelay = 0u,
      .pUserData = this,
      .pfnSequenceCallback = video_sequence_handler,
      .pfnDecodePicture = picture_decode_handler,
      .pfnDisplayPicture = picture_display_handler,
      .pfnGetOperatingPoint = operating_point_handler,
  };
  check_for_cuda_errors(
      cuvidCreateVideoParser(&parser, &parserParams), __LINE__);
}

Decoder::~Decoder() {
  if (parser) {
    cuvidDestroyVideoParser(parser);
  }
  cuvidCtxLockDestroy(ctxLock);
}

void Decoder::release() {
  cuCtxPushCurrent(cuContext);
  if (decoder) {
    cuvidDestroyDecoder(decoder);
  }
  while (!decodedFrames.empty()) {
    uint8_t* frame = decodedFrames.front();
    decodedFrames.pop();
    cuMemFree((CUdeviceptr)frame);
  }
  cuCtxPopCurrent(NULL);
}

unsigned long Decoder::decode(const uint8_t* data, unsigned long size) {
  numDecodedFrames = 0;
  CUVIDSOURCEDATAPACKET pkt = {
      .flags = CUVID_PKT_TIMESTAMP,
      .payload_size = size,
      .payload = data,
      .timestamp = 0};
  if (!data || size == 0) {
    pkt.flags |= CUVID_PKT_ENDOFSTREAM;
  }
  check_for_cuda_errors(cuvidParseVideoData(parser, &pkt), __LINE__);
  cuvidStream = 0;
  return numDecodedFrames;
}

uint8_t* Decoder::fetch_frame() {
  if (decodedFrames.empty()) {
    return nullptr;
  }
  uint8_t* frame = decodedFrames.front();
  decodedFrames.pop();
  return frame;
}

int Decoder::handle_picture_decode(CUVIDPICPARAMS* picParams) {
  if (!decoder) {
    throw std::runtime_error("Uninitialised decoder.");
  }
  picNumInDecodeOrder[picParams->CurrPicIdx] = decodePicCount++;
  check_for_cuda_errors(cuCtxPushCurrent(cuContext), __LINE__);
  check_for_cuda_errors(cuvidDecodePicture(decoder, picParams), __LINE__);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__);
  return 1;
}

int Decoder::handle_picture_display(CUVIDPARSERDISPINFO* dispInfo) {
  CUVIDPROCPARAMS procParams = {
      .progressive_frame = dispInfo->progressive_frame,
      .second_field = dispInfo->repeat_first_field + 1,
      .top_field_first = dispInfo->top_field_first,
      .unpaired_field = dispInfo->repeat_first_field < 0,
      .output_stream = cuvidStream,
  };

  CUdeviceptr dpSrcFrame = 0;
  unsigned int nSrcPitch = 0;
  check_for_cuda_errors(cuCtxPushCurrent(cuContext), __LINE__);
  check_for_cuda_errors(
      cuvidMapVideoFrame(
          decoder,
          dispInfo->picture_index,
          &dpSrcFrame,
          &nSrcPitch,
          &procParams),
      __LINE__);

  CUVIDGETDECODESTATUS decodeStatus;
  memset(&decodeStatus, 0, sizeof(decodeStatus));
  CUresult result =
      cuvidGetDecodeStatus(decoder, dispInfo->picture_index, &decodeStatus);
  if (result == CUDA_SUCCESS &&
      (decodeStatus.decodeStatus == cuvidDecodeStatus_Error ||
       decodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed)) {
    printf(
        "Decode Error occurred for picture %d\n",
        picNumInDecodeOrder[dispInfo->picture_index]);
  }

  uint8_t* decodedFrame = nullptr;
  cuMemAlloc((CUdeviceptr*)&decodedFrame, get_frame_size());

  numDecodedFrames++;

  // Copy luma plane
  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.srcDevice = dpSrcFrame;
  m.srcPitch = nSrcPitch;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = (CUdeviceptr)(m.dstHost = decodedFrame);
  m.dstPitch = get_width() * bytesPerPixel;
  m.WidthInBytes = get_width() * bytesPerPixel;
  m.Height = lumaHeight;
  check_for_cuda_errors(cuMemcpy2DAsync(&m, cuvidStream), __LINE__);

  // Copy chroma plane
  // NVDEC output has luma height aligned by 2. Adjust chroma offset by aligning
  // height
  m.srcDevice =
      (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((surfaceHeight + 1) & ~1));
  m.dstDevice =
      (CUdeviceptr)(m.dstHost = decodedFrame + m.dstPitch * lumaHeight);
  m.Height = chromaHeight;
  check_for_cuda_errors(cuMemcpy2DAsync(&m, cuvidStream), __LINE__);

  if (numChromaPlanes == 2) {
    m.srcDevice =
        (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((surfaceHeight + 1) & ~1) * 2);
    m.dstDevice =
        (CUdeviceptr)(m.dstHost = decodedFrame + m.dstPitch * lumaHeight * 2);
    m.Height = chromaHeight;
    check_for_cuda_errors(cuMemcpy2DAsync(&m, cuvidStream), __LINE__);
  }
  check_for_cuda_errors(cuStreamSynchronize(cuvidStream), __LINE__);
  decodedFrames.push(decodedFrame);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__);

  check_for_cuda_errors(cuvidUnmapVideoFrame(decoder, dpSrcFrame), __LINE__);
  return 1;
}

void Decoder::query_hardware(CUVIDEOFORMAT* videoFormat) {
  CUVIDDECODECAPS decodeCaps = {
      .eCodecType = videoFormat->codec,
      .eChromaFormat = videoFormat->chroma_format,
      .nBitDepthMinus8 = videoFormat->bit_depth_luma_minus8,
  };
  check_for_cuda_errors(cuCtxPushCurrent(cuContext), __LINE__);
  check_for_cuda_errors(cuvidGetDecoderCaps(&decodeCaps), __LINE__);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__);

  if (!decodeCaps.bIsSupported) {
    throw std::runtime_error("Codec not supported on this GPU");
  }
  if ((videoFormat->coded_width > decodeCaps.nMaxWidth) ||
      (videoFormat->coded_height > decodeCaps.nMaxHeight)) {
    std::ostringstream errorString;
    errorString << std::endl
                << "Resolution          : " << videoFormat->coded_width << "x"
                << videoFormat->coded_height << std::endl
                << "Max Supported (wxh) : " << decodeCaps.nMaxWidth << "x"
                << decodeCaps.nMaxHeight << std::endl
                << "Resolution not supported on this GPU";

    const std::string cErr = errorString.str();
    throw std::runtime_error(cErr);
  }
  if ((videoFormat->coded_width >> 4) * (videoFormat->coded_height >> 4) >
      decodeCaps.nMaxMBCount) {
    std::ostringstream errorString;
    errorString << std::endl
                << "MBCount             : "
                << (videoFormat->coded_width >> 4) *
            (videoFormat->coded_height >> 4)
                << std::endl
                << "Max Supported mbcnt : " << decodeCaps.nMaxMBCount
                << std::endl
                << "MBCount not supported on this GPU";

    const std::string cErr = errorString.str();
    throw std::runtime_error(cErr);
  }
  // Check if output format supported. If not, check fallback options
  if (!(decodeCaps.nOutputFormatMask & (1 << videoOutputFormat))) {
    if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
      videoOutputFormat = cudaVideoSurfaceFormat_NV12;
    else if (decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016))
      videoOutputFormat = cudaVideoSurfaceFormat_P016;
    else if (
        decodeCaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444))
      videoOutputFormat = cudaVideoSurfaceFormat_YUV444;
    else if (
        decodeCaps.nOutputFormatMask &
        (1 << cudaVideoSurfaceFormat_YUV444_16Bit))
      videoOutputFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
    else
      throw std::runtime_error("No supported output format found");
  }
}

int Decoder::handle_video_sequence(CUVIDEOFORMAT* vidFormat) {
  // videoCodec has been set in the constructor (for parser). Here it's set
  // again for potential correction
  videoCodec = vidFormat->codec;
  videoChromaFormat = vidFormat->chroma_format;
  bitDepthMinus8 = vidFormat->bit_depth_luma_minus8;
  bytesPerPixel = bitDepthMinus8 > 0 ? 2 : 1;
  // Set the output surface format same as chroma format
  switch (videoChromaFormat) {
    case cudaVideoChromaFormat_Monochrome:
    case cudaVideoChromaFormat_420:
      videoOutputFormat = vidFormat->bit_depth_luma_minus8
          ? cudaVideoSurfaceFormat_P016
          : cudaVideoSurfaceFormat_NV12;
      break;
    case cudaVideoChromaFormat_444:
      videoOutputFormat = vidFormat->bit_depth_luma_minus8
          ? cudaVideoSurfaceFormat_YUV444_16Bit
          : cudaVideoSurfaceFormat_YUV444;
      break;
    case cudaVideoChromaFormat_422:
      videoOutputFormat =
          cudaVideoSurfaceFormat_NV12; // no 4:2:2 output format supported yet
                                       // so make 420 default
  }

  query_hardware(vidFormat);
  if (width && lumaHeight && chromaHeight) {
    // cuvidCreateDecoder() has been called before, and now there's possible
    // config change
    return reconfigure_decoder(vidFormat);
  }

  videoFormat = *vidFormat;
  unsigned long nDecodeSurface = vidFormat->min_num_decode_surfaces;
  cudaVideoDeinterlaceMode deinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
  if (vidFormat->progressive_sequence) {
    deinterlaceMode = cudaVideoDeinterlaceMode_Weave;
  }

  CUVIDDECODECREATEINFO videoDecodeCreateInfo = {
      .ulWidth = vidFormat->coded_width,
      .ulHeight = vidFormat->coded_height,
      .ulNumDecodeSurfaces = nDecodeSurface,
      .CodecType = vidFormat->codec,
      .ChromaFormat = vidFormat->chroma_format,
      // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded
      // by NVDEC hardware
      .ulCreationFlags = cudaVideoCreate_PreferCUVID,
      .bitDepthMinus8 = vidFormat->bit_depth_luma_minus8,
      .OutputFormat = videoOutputFormat,
      .DeinterlaceMode = deinterlaceMode,
      .ulNumOutputSurfaces = 2,
      .vidLock = ctxLock,
  };
  // AV1 has max width/height of sequence in sequence header
  if (vidFormat->codec == cudaVideoCodec_AV1 &&
      vidFormat->seqhdr_data_length > 0) {
    CUVIDEOFORMATEX* vidFormatEx = (CUVIDEOFORMATEX*)vidFormat;
    maxWidth = vidFormatEx->av1.max_width;
    maxHeight = vidFormatEx->av1.max_height;
  }
  if (maxWidth < vidFormat->coded_width) {
    maxWidth = vidFormat->coded_width;
  }
  if (maxHeight < vidFormat->coded_height) {
    maxHeight = vidFormat->coded_height;
  }
  videoDecodeCreateInfo.ulMaxWidth = maxWidth;
  videoDecodeCreateInfo.ulMaxHeight = maxHeight;
  width = vidFormat->display_area.right - vidFormat->display_area.left;
  lumaHeight = vidFormat->display_area.bottom - vidFormat->display_area.top;
  videoDecodeCreateInfo.ulTargetWidth = vidFormat->coded_width;
  videoDecodeCreateInfo.ulTargetHeight = vidFormat->coded_height;
  chromaHeight =
      (int)(ceil(lumaHeight * chroma_height_factor(videoOutputFormat)));
  numChromaPlanes = chroma_plane_count(videoOutputFormat);
  surfaceHeight = videoDecodeCreateInfo.ulTargetHeight;
  surfaceWidth = videoDecodeCreateInfo.ulTargetWidth;
  displayRect.bottom = videoDecodeCreateInfo.display_area.bottom;
  displayRect.top = videoDecodeCreateInfo.display_area.top;
  displayRect.left = videoDecodeCreateInfo.display_area.left;
  displayRect.right = videoDecodeCreateInfo.display_area.right;

  check_for_cuda_errors(cuCtxPushCurrent(cuContext), __LINE__);
  check_for_cuda_errors(
      cuvidCreateDecoder(&decoder, &videoDecodeCreateInfo), __LINE__);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__);
  return nDecodeSurface;
}

int Decoder::reconfigure_decoder(CUVIDEOFORMAT* vidFormat) {
  if (vidFormat->bit_depth_luma_minus8 != videoFormat.bit_depth_luma_minus8 ||
      vidFormat->bit_depth_chroma_minus8 !=
          videoFormat.bit_depth_chroma_minus8) {
    throw std::runtime_error("Reconfigure Not supported for bit depth change");
  }
  if (vidFormat->chroma_format != videoFormat.chroma_format) {
    throw std::runtime_error(
        "Reconfigure Not supported for chroma format change");
  }

  bool bDecodeResChange =
      !(vidFormat->coded_width == videoFormat.coded_width &&
        vidFormat->coded_height == videoFormat.coded_height);
  bool bDisplayRectChange =
      !(vidFormat->display_area.bottom == videoFormat.display_area.bottom &&
        vidFormat->display_area.top == videoFormat.display_area.top &&
        vidFormat->display_area.left == videoFormat.display_area.left &&
        vidFormat->display_area.right == videoFormat.display_area.right);

  int nDecodeSurface = vidFormat->min_num_decode_surfaces;

  if ((vidFormat->coded_width > maxWidth) ||
      (vidFormat->coded_height > maxHeight)) {
    // For VP9, let driver  handle the change if new width/height >
    // maxwidth/maxheight
    if ((videoCodec != cudaVideoCodec_VP9) || reconfigExternal) {
      throw std::runtime_error(
          "Reconfigure Not supported when width/height > maxwidth/maxheight");
    }
    return 1;
  }

  if (!bDecodeResChange && !reconfigExtPPChange) {
    // if the coded_width/coded_height hasn't changed but display resolution has
    // changed, then need to update width/height for correct output without
    // cropping. Example : 1920x1080 vs 1920x1088
    if (bDisplayRectChange) {
      width = vidFormat->display_area.right - vidFormat->display_area.left;
      lumaHeight = vidFormat->display_area.bottom - vidFormat->display_area.top;
      chromaHeight =
          (int)ceil(lumaHeight * chroma_height_factor(videoOutputFormat));
      numChromaPlanes = chroma_plane_count(videoOutputFormat);
    }

    // no need for reconfigureDecoder(). Just return
    return 1;
  }

  CUVIDRECONFIGUREDECODERINFO reconfigParams = {0};

  reconfigParams.ulWidth = videoFormat.coded_width = vidFormat->coded_width;
  reconfigParams.ulHeight = videoFormat.coded_height = vidFormat->coded_height;

  // Dont change display rect and get scaled output from decoder. This will help
  // display app to present apps smoothly
  reconfigParams.display_area.bottom = displayRect.bottom;
  reconfigParams.display_area.top = displayRect.top;
  reconfigParams.display_area.left = displayRect.left;
  reconfigParams.display_area.right = displayRect.right;
  reconfigParams.ulTargetWidth = surfaceWidth;
  reconfigParams.ulTargetHeight = surfaceHeight;

  // If external reconfigure is called along with resolution change even if post
  // processing params is not changed, do full reconfigure params update
  if ((reconfigExternal && bDecodeResChange) || reconfigExtPPChange) {
    // update display rect and target resolution if requested explicitely
    reconfigExternal = false;
    reconfigExtPPChange = false;
    videoFormat = *vidFormat;
    width = vidFormat->display_area.right - vidFormat->display_area.left;
    lumaHeight = vidFormat->display_area.bottom - vidFormat->display_area.top;
    reconfigParams.ulTargetWidth = vidFormat->coded_width;
    reconfigParams.ulTargetHeight = vidFormat->coded_height;
    chromaHeight =
        (int)ceil(lumaHeight * chroma_height_factor(videoOutputFormat));
    numChromaPlanes = chroma_plane_count(videoOutputFormat);
    surfaceHeight = reconfigParams.ulTargetHeight;
    surfaceWidth = reconfigParams.ulTargetWidth;
    displayRect.bottom = reconfigParams.display_area.bottom;
    displayRect.top = reconfigParams.display_area.top;
    displayRect.left = reconfigParams.display_area.left;
    displayRect.right = reconfigParams.display_area.right;
  }
  reconfigParams.ulNumDecodeSurfaces = nDecodeSurface;

  check_for_cuda_errors(cuCtxPushCurrent(cuContext), __LINE__);
  check_for_cuda_errors(
      cuvidReconfigureDecoder(decoder, &reconfigParams), __LINE__);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__);

  return nDecodeSurface;
}

int Decoder::get_operating_point(CUVIDOPERATINGPOINTINFO* operPointInfo) {
  if (operPointInfo->codec == cudaVideoCodec_AV1) {
    if (operPointInfo->av1.operating_points_cnt > 1) {
      // clip has SVC enabled
      if (operatingPoint >= operPointInfo->av1.operating_points_cnt)
        operatingPoint = 0;

      printf(
          "AV1 SVC clip: operating point count %d  ",
          operPointInfo->av1.operating_points_cnt);
      printf(
          "Selected operating point: %d, IDC 0x%x bOutputAllLayers %d\n",
          operatingPoint,
          operPointInfo->av1.operating_points_idc[operatingPoint],
          dispAllLayers);
      return (operatingPoint | (dispAllLayers << 10));
    }
  }
  return -1;
}
