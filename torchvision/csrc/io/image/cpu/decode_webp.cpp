#include "decode_webp.h"
#include "webp/decode.h"

namespace vision {
namespace image {

// TODO IF WEBP_FOUND etc.

torch::Tensor decode_webp(const torch::Tensor& data) {
    int width = 0;
    int height = 0;

    auto out_data = WebPDecodeRGB(data.data_ptr<uint8_t>(), data.numel(), &width, &height);

    printf("width: %d, height: %d\n", width, height);
    auto out = torch::from_blob(out_data, {height, width, 3}, torch::kUInt8);
    return out.permute({2, 0, 1});
}

} // namespace image
} // namespace vision