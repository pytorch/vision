#include "decode_jpeg.h"
#include "../common.h"
#include "common_jpeg.h"
#include "exif.h"

#include <c10/util/Optional.h>

namespace vision {
namespace image {

#if !JPEG_FOUND
torch::Tensor decode_jpeg(
    const torch::Tensor& data,
    ImageReadMode mode,
    bool apply_exif_orientation) {
  STD_TORCH_CHECK(
      false, "decode_jpeg: torchvision not compiled with libjpeg support");
}
#else

using namespace detail;
using namespace exif_private;

namespace {

struct torch_jpeg_mgr {
  struct jpeg_source_mgr pub;
  const JOCTET* data;
  size_t len;
};

static void torch_jpeg_init_source(j_decompress_ptr cinfo) {}

static boolean torch_jpeg_fill_input_buffer(j_decompress_ptr cinfo) {
  // No more data.  Probably an incomplete image;  Raise exception.
  torch_jpeg_error_ptr myerr = (torch_jpeg_error_ptr)cinfo->err;
  strcpy(myerr->jpegLastErrorMsg, "Image is incomplete or truncated");
  longjmp(myerr->setjmp_buffer, 1);
}

static void torch_jpeg_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
  torch_jpeg_mgr* src = (torch_jpeg_mgr*)cinfo->src;
  if (src->pub.bytes_in_buffer < (size_t)num_bytes) {
    // Skipping over all of remaining data;  output EOI.
    src->pub.next_input_byte = EOI_BUFFER;
    src->pub.bytes_in_buffer = 1;
  } else {
    // Skipping over only some of the remaining data.
    src->pub.next_input_byte += num_bytes;
    src->pub.bytes_in_buffer -= num_bytes;
  }
}

static void torch_jpeg_term_source(j_decompress_ptr cinfo) {}

static void torch_jpeg_set_source_mgr(
    j_decompress_ptr cinfo,
    const unsigned char* data,
    size_t len) {
  torch_jpeg_mgr* src;
  if (cinfo->src == 0) { // if this is first time;  allocate memory
    cinfo->src = (struct jpeg_source_mgr*)(*cinfo->mem->alloc_small)(
        (j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(torch_jpeg_mgr));
  }
  src = (torch_jpeg_mgr*)cinfo->src;
  src->pub.init_source = torch_jpeg_init_source;
  src->pub.fill_input_buffer = torch_jpeg_fill_input_buffer;
  src->pub.skip_input_data = torch_jpeg_skip_input_data;
  src->pub.resync_to_restart = jpeg_resync_to_restart; // default
  src->pub.term_source = torch_jpeg_term_source;
  // fill the buffers
  src->data = (const JOCTET*)data;
  src->len = len;
  src->pub.bytes_in_buffer = len;
  src->pub.next_input_byte = src->data;

  jpeg_save_markers(cinfo, APP1, 0xffff);
}

inline unsigned char clamped_cmyk_rgb_convert(
    unsigned char k,
    unsigned char cmy) {
  int v = k * cmy + 128;
  v = ((v >> 8) + v) >> 8;
  return std::clamp(k - v, 0, 255);
}

void convert_line_cmyk_to_rgb(
    j_decompress_ptr cinfo,
    const unsigned char* cmyk_line,
    unsigned char* rgb_line) {
  int width = cinfo->output_width;
  for (int i = 0; i < width; ++i) {
    int c = cmyk_line[i * 4 + 0];
    int m = cmyk_line[i * 4 + 1];
    int y = cmyk_line[i * 4 + 2];
    int k = cmyk_line[i * 4 + 3];

    rgb_line[i * 3 + 0] = clamped_cmyk_rgb_convert(k, 255 - c);
    rgb_line[i * 3 + 1] = clamped_cmyk_rgb_convert(k, 255 - m);
    rgb_line[i * 3 + 2] = clamped_cmyk_rgb_convert(k, 255 - y);
  }
}

inline unsigned char rgb_to_gray(int r, int g, int b) {
  return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
}

void convert_line_cmyk_to_gray(
    j_decompress_ptr cinfo,
    const unsigned char* cmyk_line,
    unsigned char* gray_line) {
  int width = cinfo->output_width;
  for (int i = 0; i < width; ++i) {
    int c = cmyk_line[i * 4 + 0];
    int m = cmyk_line[i * 4 + 1];
    int y = cmyk_line[i * 4 + 2];
    int k = cmyk_line[i * 4 + 3];

    int r = clamped_cmyk_rgb_convert(k, 255 - c);
    int g = clamped_cmyk_rgb_convert(k, 255 - m);
    int b = clamped_cmyk_rgb_convert(k, 255 - y);

    gray_line[i] = rgb_to_gray(r, g, b);
  }
}

// Helper return type: keep longjmp/libjpeg contained here.
struct JpegDecodeResult {
  torch::Tensor hwc;         // HWC uint8 tensor
  int exif_orientation = -1; // valid if apply_exif_orientation==true
};

// Decode to HWC only. No permute/exif transform here (keep throwing ops outside longjmp zone).
static JpegDecodeResult decode_jpeg_hwc_impl(
    const torch::Tensor& data,
    ImageReadMode mode,
    bool apply_exif_orientation) {
  JpegDecodeResult res;

  validate_encoded_data(data);

  struct jpeg_decompress_struct cinfo;
  struct torch_jpeg_error_mgr jerr;

  // NOTE: libjpeg uses setjmp/longjmp. longjmp does not unwind C++ stack frames.
  // Any tensors allocated after setjmp must be explicitly reset in the error path.
  c10::optional<torch::Tensor> tensor_opt;
  c10::optional<torch::Tensor> cmyk_line_opt;

  auto datap = data.data_ptr<uint8_t>();
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = torch_jpeg_error_exit;

  if (setjmp(jerr.setjmp_buffer)) {
    cmyk_line_opt.reset();
    tensor_opt.reset();
    jpeg_destroy_decompress(&cinfo);
    STD_TORCH_CHECK(false, jerr.jpegLastErrorMsg);
  }

  jpeg_create_decompress(&cinfo);
  torch_jpeg_set_source_mgr(&cinfo, datap, data.numel());

  jpeg_read_header(&cinfo, TRUE);

  int channels = cinfo.num_components;
  bool cmyk_to_rgb_or_gray = false;

  if (mode != IMAGE_READ_MODE_UNCHANGED) {
    switch (mode) {
      case IMAGE_READ_MODE_GRAY:
        if (cinfo.jpeg_color_space == JCS_CMYK ||
            cinfo.jpeg_color_space == JCS_YCCK) {
          cinfo.out_color_space = JCS_CMYK;
          cmyk_to_rgb_or_gray = true;
        } else {
          cinfo.out_color_space = JCS_GRAYSCALE;
        }
        channels = 1;
        break;
      case IMAGE_READ_MODE_RGB:
        if (cinfo.jpeg_color_space == JCS_CMYK ||
            cinfo.jpeg_color_space == JCS_YCCK) {
          cinfo.out_color_space = JCS_CMYK;
          cmyk_to_rgb_or_gray = true;
        } else {
          cinfo.out_color_space = JCS_RGB;
        }
        channels = 3;
        break;
      default:
        jpeg_destroy_decompress(&cinfo);
        STD_TORCH_CHECK(false, "The provided mode is not supported for JPEG files");
    }
    jpeg_calc_output_dimensions(&cinfo);
  }

  if (apply_exif_orientation) {
    res.exif_orientation = fetch_jpeg_exif_orientation(&cinfo);
  }

  jpeg_start_decompress(&cinfo);

  int height = cinfo.output_height;
  int width = cinfo.output_width;

  int stride = width * channels;

  tensor_opt =
      torch::empty({int64_t(height), int64_t(width), channels}, torch::kU8);
  auto ptr = tensor_opt->data_ptr<uint8_t>();

  if (cmyk_to_rgb_or_gray) {
    cmyk_line_opt = torch::empty({int64_t(width), 4}, torch::kU8);
  }

  while (cinfo.output_scanline < cinfo.output_height) {
    if (cmyk_to_rgb_or_gray) {
      auto cmyk_line_ptr = cmyk_line_opt->data_ptr<uint8_t>();
      jpeg_read_scanlines(&cinfo, &cmyk_line_ptr, 1);

      if (channels == 3) {
        convert_line_cmyk_to_rgb(&cinfo, cmyk_line_ptr, ptr);
      } else if (channels == 1) {
        convert_line_cmyk_to_gray(&cinfo, cmyk_line_ptr, ptr);
      }
    } else {
      jpeg_read_scanlines(&cinfo, &ptr, 1);
    }
    ptr += stride;
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  res.hwc = *tensor_opt;
  return res;
}

} // namespace

torch::Tensor decode_jpeg(
    const torch::Tensor& data,
    ImageReadMode mode,
    bool apply_exif_orientation) {
  C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cpu.decode_jpeg.decode_jpeg");

  // Longjmp/libjpeg zone is inside helper.
  auto res = decode_jpeg_hwc_impl(data, mode, apply_exif_orientation);

  // Throwing ops are outside the longjmp zone.
  auto output = res.hwc.permute({2, 0, 1});
  if (apply_exif_orientation) {
    return exif_orientation_transform(output, res.exif_orientation);
  }
  return output;
}

#endif // #if !JPEG_FOUND

int64_t _jpeg_version() {
#if JPEG_FOUND
  return JPEG_LIB_VERSION;
#else
  return -1;
#endif
}

bool _is_compiled_against_turbo() {
#ifdef LIBJPEG_TURBO_VERSION
  return true;
#else
  return false;
#endif
}

} // namespace image
} // namespace vision