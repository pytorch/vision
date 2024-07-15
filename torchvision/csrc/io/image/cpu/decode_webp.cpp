#include "decode_webp.h"
#include "webp/decode.h"

namespace vision {
namespace image {

// TODO IF WEBP_FOUND etc.

torch::Tensor decode_webp(const torch::Tensor& data) {
    int width = 0;
    int height = 0;

    if (!WebPGetInfo(data.data_ptr<uint8_t>(), data.numel(), &width, &height)) {
        TORCH_CHECK(false, "WebPGetInfo failed");
    }
    auto out = torch::empty({height, width, 3}, torch::kUInt8);
    WebPDecodeRGBInto(data.data_ptr<uint8_t>(), data.numel(), out.data_ptr<uint8_t>(), out.numel(), /*output_stride=*/3 * width);
    return out.permute({2, 0, 1});  // return CHW, channels-last
}

} // namespace image
} // namespace vision