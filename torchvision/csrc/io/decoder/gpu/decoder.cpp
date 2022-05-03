#include "decoder.h"
#include <c10/util/Logging.h>
#include <nppi_color_conversion.h>
#include <cmath>
#include <cstring>
#include <unordered_map>

static float chroma_height_factor(cudaVideoSurfaceFormat surface_format) {
  return (surface_format == cudaVideoSurfaceFormat_YUV444 ||
          surface_format == cudaVideoSurfaceFormat_YUV444_16Bit)
      ? 1.0
      : 0.5;
}

static int chroma_plane_count(cudaVideoSurfaceFormat surface_format) {
  return (surface_format == cudaVideoSurfaceFormat_YUV444 ||
          surface_format == cudaVideoSurfaceFormat_YUV444_16Bit)
      ? 2
      : 1;
}

/* Initialise cu_context and video_codec, create context lock and create parser
 * object.
 */
void Decoder::init(CUcontext context, cudaVideoCodec codec) {
  cu_context = context;
  video_codec = codec;
  check_for_cuda_errors(
      cuvidCtxLockCreate(&ctx_lock, cu_context), __LINE__, __FILE__);

  CUVIDPARSERPARAMS parser_params = {};
  parser_params.CodecType = codec;
  parser_params.ulMaxNumDecodeSurfaces = 1;
  parser_params.ulClockRate = 1000;
  parser_params.ulMaxDisplayDelay = 0u;
  parser_params.pUserData = this;
  parser_params.pfnSequenceCallback = video_sequence_handler;
  parser_params.pfnDecodePicture = picture_decode_handler;
  parser_params.pfnDisplayPicture = picture_display_handler;
  parser_params.pfnGetOperatingPoint = operating_point_handler;

  check_for_cuda_errors(
      cuvidCreateVideoParser(&parser, &parser_params), __LINE__, __FILE__);
}

/* Destroy parser object and context lock.
 */
Decoder::~Decoder() {
  if (parser) {
    cuvidDestroyVideoParser(parser);
  }
  cuvidCtxLockDestroy(ctx_lock);
}

/* Destroy CUvideodecoder object and free up all the unreturned decoded frames.
 */
void Decoder::release() {
  cuCtxPushCurrent(cu_context);
  if (decoder) {
    cuvidDestroyDecoder(decoder);
  }
  cuCtxPopCurrent(NULL);
}

/* Trigger video decoding.
 */
void Decoder::decode(const uint8_t* data, unsigned long size) {
  CUVIDSOURCEDATAPACKET pkt = {};
  pkt.flags = CUVID_PKT_TIMESTAMP;
  pkt.payload_size = size;
  pkt.payload = data;
  pkt.timestamp = 0;
  if (!data || size == 0) {
    pkt.flags |= CUVID_PKT_ENDOFSTREAM;
  }
  check_for_cuda_errors(cuvidParseVideoData(parser, &pkt), __LINE__, __FILE__);
  cuvidStream = 0;
}

/* Fetch a decoded frame and remove it from the queue.
 */
torch::Tensor Decoder::fetch_frame() {
  if (decoded_frames.empty()) {
    auto options =
        torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
    return torch::zeros({0}, options);
  }
  torch::Tensor frame = decoded_frames.front();
  decoded_frames.pop();
  return frame;
}

/* Called when a picture is ready to be decoded.
 */
int Decoder::handle_picture_decode(CUVIDPICPARAMS* pic_params) {
  if (!decoder) {
    TORCH_CHECK(false, "Uninitialised decoder");
  }
  pic_num_in_decode_order[pic_params->CurrPicIdx] = decode_pic_count++;
  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
  check_for_cuda_errors(
      cuvidDecodePicture(decoder, pic_params), __LINE__, __FILE__);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__, __FILE__);
  return 1;
}

/* Process the decoded data and copy it to a cuda memory location.
 */
int Decoder::handle_picture_display(CUVIDPARSERDISPINFO* disp_info) {
  CUVIDPROCPARAMS proc_params = {};
  proc_params.progressive_frame = disp_info->progressive_frame;
  proc_params.second_field = disp_info->repeat_first_field + 1;
  proc_params.top_field_first = disp_info->top_field_first;
  proc_params.unpaired_field = disp_info->repeat_first_field < 0;
  proc_params.output_stream = cuvidStream;

  CUdeviceptr source_frame = 0;
  unsigned int source_pitch = 0;
  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
  check_for_cuda_errors(
      cuvidMapVideoFrame(
          decoder,
          disp_info->picture_index,
          &source_frame,
          &source_pitch,
          &proc_params),
      __LINE__,
      __FILE__);

  CUVIDGETDECODESTATUS decode_status;
  memset(&decode_status, 0, sizeof(decode_status));
  CUresult result =
      cuvidGetDecodeStatus(decoder, disp_info->picture_index, &decode_status);
  if (result == CUDA_SUCCESS &&
      (decode_status.decodeStatus == cuvidDecodeStatus_Error ||
       decode_status.decodeStatus == cuvidDecodeStatus_Error_Concealed)) {
    VLOG(1) << "Decode Error occurred for picture "
            << pic_num_in_decode_order[disp_info->picture_index];
  }

  auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
  torch::Tensor decoded_frame = torch::empty({get_height(), width, 3}, options);
  uint8_t* frame_ptr = decoded_frame.data_ptr<uint8_t>();
  const uint8_t* const source_arr[] = {
      (const uint8_t* const)source_frame,
      (const uint8_t* const)(source_frame + source_pitch * ((surface_height + 1) & ~1))};

  auto err = nppiNV12ToRGB_709CSC_8u_P2C3R(
      source_arr,
      source_pitch,
      frame_ptr,
      width * 3,
      {(int)decoded_frame.size(1), (int)decoded_frame.size(0)});

  TORCH_CHECK(
      err == NPP_NO_ERROR,
      "Failed to convert from NV12 to RGB. Error code:",
      err);

  check_for_cuda_errors(cuStreamSynchronize(cuvidStream), __LINE__, __FILE__);
  decoded_frames.push(decoded_frame);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__, __FILE__);

  check_for_cuda_errors(
      cuvidUnmapVideoFrame(decoder, source_frame), __LINE__, __FILE__);
  return 1;
}

/* Query the capabilities of the underlying hardware video decoder and
 * verify if the hardware supports decoding the passed video.
 */
void Decoder::query_hardware(CUVIDEOFORMAT* video_format) {
  CUVIDDECODECAPS decode_caps = {};
  decode_caps.eCodecType = video_format->codec;
  decode_caps.eChromaFormat = video_format->chroma_format;
  decode_caps.nBitDepthMinus8 = video_format->bit_depth_luma_minus8;

  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
  check_for_cuda_errors(cuvidGetDecoderCaps(&decode_caps), __LINE__, __FILE__);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__, __FILE__);

  if (!decode_caps.bIsSupported) {
    TORCH_CHECK(false, "Codec not supported on this GPU");
  }
  if ((video_format->coded_width > decode_caps.nMaxWidth) ||
      (video_format->coded_height > decode_caps.nMaxHeight)) {
    TORCH_CHECK(
        false,
        "Resolution          : ",
        video_format->coded_width,
        "x",
        video_format->coded_height,
        "\nMax Supported (wxh) : ",
        decode_caps.nMaxWidth,
        "x",
        decode_caps.nMaxHeight,
        "\nResolution not supported on this GPU");
  }
  if ((video_format->coded_width >> 4) * (video_format->coded_height >> 4) >
      decode_caps.nMaxMBCount) {
    TORCH_CHECK(
        false,
        "MBCount             : ",
        (video_format->coded_width >> 4) * (video_format->coded_height >> 4),
        "\nMax Supported mbcnt : ",
        decode_caps.nMaxMBCount,
        "\nMBCount not supported on this GPU");
  }
  // Check if output format supported. If not, check fallback options
  if (!(decode_caps.nOutputFormatMask & (1 << video_output_format))) {
    if (decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12)) {
      video_output_format = cudaVideoSurfaceFormat_NV12;
    } else if (
        decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016)) {
      video_output_format = cudaVideoSurfaceFormat_P016;
    } else if (
        decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444)) {
      video_output_format = cudaVideoSurfaceFormat_YUV444;
    } else if (
        decode_caps.nOutputFormatMask &
        (1 << cudaVideoSurfaceFormat_YUV444_16Bit)) {
      video_output_format = cudaVideoSurfaceFormat_YUV444_16Bit;
    } else {
      TORCH_CHECK(false, "No supported output format found");
    }
  }
}

/* Called before decoding frames and/or whenever there is a configuration
 * change.
 */
int Decoder::handle_video_sequence(CUVIDEOFORMAT* video_format) {
  // video_codec has been set in init(). Here it's set
  // again for potential correction.
  video_codec = video_format->codec;
  video_chroma_format = video_format->chroma_format;
  bit_depth_minus8 = video_format->bit_depth_luma_minus8;
  bytes_per_pixel = bit_depth_minus8 > 0 ? 2 : 1;
  // Set the output surface format same as chroma format
  switch (video_chroma_format) {
    case cudaVideoChromaFormat_Monochrome:
    case cudaVideoChromaFormat_420:
      video_output_format = video_format->bit_depth_luma_minus8
          ? cudaVideoSurfaceFormat_P016
          : cudaVideoSurfaceFormat_NV12;
      break;
    case cudaVideoChromaFormat_444:
      video_output_format = video_format->bit_depth_luma_minus8
          ? cudaVideoSurfaceFormat_YUV444_16Bit
          : cudaVideoSurfaceFormat_YUV444;
      break;
    case cudaVideoChromaFormat_422:
      video_output_format = cudaVideoSurfaceFormat_NV12;
  }

  query_hardware(video_format);

  if (width && luma_height && chroma_height) {
    // cuvidCreateDecoder() has been called before and now there's possible
    // config change.
    return reconfigure_decoder(video_format);
  }

  cu_video_format = *video_format;
  unsigned long decode_surface = video_format->min_num_decode_surfaces;
  cudaVideoDeinterlaceMode deinterlace_mode = cudaVideoDeinterlaceMode_Adaptive;

  if (video_format->progressive_sequence) {
    deinterlace_mode = cudaVideoDeinterlaceMode_Weave;
  }

  CUVIDDECODECREATEINFO video_decode_create_info = {};
  video_decode_create_info.ulWidth = video_format->coded_width;
  video_decode_create_info.ulHeight = video_format->coded_height;
  video_decode_create_info.ulNumDecodeSurfaces = decode_surface;
  video_decode_create_info.CodecType = video_format->codec;
  video_decode_create_info.ChromaFormat = video_format->chroma_format;
  // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded
  // by NVDEC hardware
  video_decode_create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  video_decode_create_info.bitDepthMinus8 = video_format->bit_depth_luma_minus8;
  video_decode_create_info.OutputFormat = video_output_format;
  video_decode_create_info.DeinterlaceMode = deinterlace_mode;
  video_decode_create_info.ulNumOutputSurfaces = 2;
  video_decode_create_info.vidLock = ctx_lock;

  // AV1 has max width/height of sequence in sequence header
  if (video_format->codec == cudaVideoCodec_AV1 &&
      video_format->seqhdr_data_length > 0) {
    CUVIDEOFORMATEX* video_format_ex = (CUVIDEOFORMATEX*)video_format;
    max_width = video_format_ex->av1.max_width;
    max_height = video_format_ex->av1.max_height;
  }
  if (max_width < video_format->coded_width) {
    max_width = video_format->coded_width;
  }
  if (max_height < video_format->coded_height) {
    max_height = video_format->coded_height;
  }
  video_decode_create_info.ulMaxWidth = max_width;
  video_decode_create_info.ulMaxHeight = max_height;
  width = video_format->display_area.right - video_format->display_area.left;
  luma_height =
      video_format->display_area.bottom - video_format->display_area.top;
  video_decode_create_info.ulTargetWidth = video_format->coded_width;
  video_decode_create_info.ulTargetHeight = video_format->coded_height;
  chroma_height =
      (int)(ceil(luma_height * chroma_height_factor(video_output_format)));
  num_chroma_planes = chroma_plane_count(video_output_format);
  surface_height = video_decode_create_info.ulTargetHeight;
  surface_width = video_decode_create_info.ulTargetWidth;
  display_rect.bottom = video_decode_create_info.display_area.bottom;
  display_rect.top = video_decode_create_info.display_area.top;
  display_rect.left = video_decode_create_info.display_area.left;
  display_rect.right = video_decode_create_info.display_area.right;

  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
  check_for_cuda_errors(
      cuvidCreateDecoder(&decoder, &video_decode_create_info),
      __LINE__,
      __FILE__);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__, __FILE__);
  return decode_surface;
}

int Decoder::reconfigure_decoder(CUVIDEOFORMAT* video_format) {
  if (video_format->bit_depth_luma_minus8 !=
          cu_video_format.bit_depth_luma_minus8 ||
      video_format->bit_depth_chroma_minus8 !=
          cu_video_format.bit_depth_chroma_minus8) {
    TORCH_CHECK(false, "Reconfigure not supported for bit depth change");
  }
  if (video_format->chroma_format != cu_video_format.chroma_format) {
    TORCH_CHECK(false, "Reconfigure not supported for chroma format change");
  }

  bool decode_res_change =
      !(video_format->coded_width == cu_video_format.coded_width &&
        video_format->coded_height == cu_video_format.coded_height);
  bool display_rect_change =
      !(video_format->display_area.bottom ==
            cu_video_format.display_area.bottom &&
        video_format->display_area.top == cu_video_format.display_area.top &&
        video_format->display_area.left == cu_video_format.display_area.left &&
        video_format->display_area.right == cu_video_format.display_area.right);

  unsigned int decode_surface = video_format->min_num_decode_surfaces;

  if ((video_format->coded_width > max_width) ||
      (video_format->coded_height > max_height)) {
    // For VP9, let driver  handle the change if new width/height >
    // maxwidth/maxheight
    if (video_codec != cudaVideoCodec_VP9) {
      TORCH_CHECK(
          false,
          "Reconfigure not supported when width/height > maxwidth/maxheight");
    }
    return 1;
  }

  if (!decode_res_change) {
    // If the coded_width/coded_height hasn't changed but display resolution has
    // changed, then need to update width/height for correct output without
    // cropping. Example : 1920x1080 vs 1920x1088.
    if (display_rect_change) {
      width =
          video_format->display_area.right - video_format->display_area.left;
      luma_height =
          video_format->display_area.bottom - video_format->display_area.top;
      chroma_height =
          (int)ceil(luma_height * chroma_height_factor(video_output_format));
      num_chroma_planes = chroma_plane_count(video_output_format);
    }
    return 1;
  }
  cu_video_format.coded_width = video_format->coded_width;
  cu_video_format.coded_height = video_format->coded_height;
  CUVIDRECONFIGUREDECODERINFO reconfig_params = {};
  reconfig_params.ulWidth = video_format->coded_width;
  reconfig_params.ulHeight = video_format->coded_height;
  reconfig_params.ulTargetWidth = surface_width;
  reconfig_params.ulTargetHeight = surface_height;
  reconfig_params.ulNumDecodeSurfaces = decode_surface;
  reconfig_params.display_area.bottom = display_rect.bottom;
  reconfig_params.display_area.top = display_rect.top;
  reconfig_params.display_area.left = display_rect.left;
  reconfig_params.display_area.right = display_rect.right;

  check_for_cuda_errors(cuCtxPushCurrent(cu_context), __LINE__, __FILE__);
  check_for_cuda_errors(
      cuvidReconfigureDecoder(decoder, &reconfig_params), __LINE__, __FILE__);
  check_for_cuda_errors(cuCtxPopCurrent(NULL), __LINE__, __FILE__);

  return decode_surface;
}

/* Called from AV1 sequence header to get operating point of an AV1 bitstream.
 */
int Decoder::get_operating_point(CUVIDOPERATINGPOINTINFO* oper_point_info) {
  return oper_point_info->codec == cudaVideoCodec_AV1 &&
          oper_point_info->av1.operating_points_cnt > 1
      ? 0
      : -1;
}
