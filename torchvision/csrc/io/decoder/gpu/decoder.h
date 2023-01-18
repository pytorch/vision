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
  int get_height() const {
    return luma_height;
  }

 private:
  unsigned int width = 0, luma_height = 0, chroma_height = 0;
  unsigned int surface_height = 0, surface_width = 0;
  unsigned int max_width = 0, max_height = 0;
  unsigned int num_chroma_planes = 0;
  int bit_depth_minus8 = 0, bytes_per_pixel = 1;
  int decode_pic_count = 0, pic_num_in_decode_order[32];
  std::queue<torch::Tensor> decoded_frames;
  CUcontext cu_context = NULL;
  CUvideoctxlock ctx_lock;
  CUvideoparser parser = NULL;
  CUvideodecoder decoder = NULL;
  CUstream cuvidStream = 0;
  cudaVideoCodec video_codec = cudaVideoCodec_NumCodecs;
  cudaVideoChromaFormat video_chroma_format = cudaVideoChromaFormat_420;
  cudaVideoSurfaceFormat video_output_format = cudaVideoSurfaceFormat_NV12;
  CUVIDEOFORMAT cu_video_format = {};
  Rect display_rect = {};

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
