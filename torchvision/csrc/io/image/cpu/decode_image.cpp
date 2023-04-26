#include "decode_image.h"

#include "decode_jpeg.h"
#include "decode_png.h"

namespace vision {
namespace image {

torch::Tensor decode_image(const torch::Tensor& data, ImageReadMode mode) {
  // Check that tensor is a CPU tensor
  TORCH_CHECK(data.device() == torch::kCPU, "Expected a CPU tensor");
  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
  // Check that the input tensor is 1-dimensional
  TORCH_CHECK(
      data.dim() == 1 && data.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  auto datap = data.data_ptr<uint8_t>();

  const uint8_t jpeg_signature[3] = {255, 216, 255}; // == "\xFF\xD8\xFF"
  const uint8_t png_signature[4] = {137, 80, 78, 71}; // == "\211PNG"

  if (memcmp(jpeg_signature, datap, 3) == 0) {
    return decode_jpeg(data, mode);
  } else if (memcmp(png_signature, datap, 4) == 0) {
    return decode_png(data, mode);
  } else {
    TORCH_CHECK(
        false,
        "Unsupported image file. Only jpeg and png ",
        "are currently supported.");
  }
}

} // namespace image
} // namespace vision
