#pragma once

#include <torch/types.h>  // TODO: Is this where C10_EXPORT is defined?? remove/change?

namespace vision {
namespace image {

C10_EXPORT void decode_gif();


} // namespace image
} // namespace vision
