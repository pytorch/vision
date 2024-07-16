#include "decode_webp.h"
#include "webp/decode.h"

namespace vision {
namespace image {

#if !WEBP_FOUND
torch::Tensor decode_webp(const torch::Tensor& data) {
  TORCH_CHECK(
      false, "decode_webp: torchvision not compiled with libwebp support");
}
#else

torch::Tensor decode_webp(const torch::Tensor& data) {
    int width = 0;
    int height = 0;

    auto decoded_data = WebPDecodeRGB(data.data_ptr<uint8_t>(), data.numel(), &width, &height);
    TORCH_CHECK(decoded_data != nullptr, "WebPDecodeRGB failed.");
    auto out = torch::from_blob(decoded_data, {height, width, 3}, torch::kUInt8);
    return out.permute({2, 0, 1});  // return CHW, channels-last
}
#endif // WEBP_FOUND

} // namespace image
} // namespace vision
