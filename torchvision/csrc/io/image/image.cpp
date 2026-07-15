#include "image.h"

#include <ATen/core/op_registration/op_registration.h>

namespace vision {
namespace image {

static auto registry = torch::RegisterOperators().op(
    "image::decode_webp(Tensor encoded_data, int mode) -> Tensor",
    &decode_webp);

} // namespace image
} // namespace vision
