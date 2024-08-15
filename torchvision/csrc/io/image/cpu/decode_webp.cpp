#include "decode_webp.h"

#if WEBP_FOUND
#include "webp/decode.h"
#endif // WEBP_FOUND

namespace vision {
namespace image {

#if !WEBP_FOUND
torch::Tensor decode_webp(const torch::Tensor& data) {
  TORCH_CHECK(
      false, "decode_webp: torchvision not compiled with libwebp support");
}
#else

torch::Tensor decode_webp(const torch::Tensor& encoded_data) {
  TORCH_CHECK(encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  TORCH_CHECK(
      encoded_data.dtype() == torch::kU8,
      "Input tensor must have uint8 data type, got ",
      encoded_data.dtype());
  TORCH_CHECK(
      encoded_data.dim() == 1,
      "Input tensor must be 1-dimensional, got ",
      encoded_data.dim(),
      " dims.");

  int width = 0;
  int height = 0;
  auto decoded_data = WebPDecodeRGB(
      encoded_data.data_ptr<uint8_t>(), encoded_data.numel(), &width, &height);
  TORCH_CHECK(decoded_data != nullptr, "WebPDecodeRGB failed.");
  auto out = torch::from_blob(decoded_data, {height, width, 3}, torch::kUInt8);
  return out.permute({2, 0, 1}); // return CHW, channels-last
}
#endif // WEBP_FOUND

} // namespace image
} // namespace vision
