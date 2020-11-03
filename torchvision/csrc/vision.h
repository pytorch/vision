#ifndef VISION_H
#define VISION_H

#include <torchvision/models/models.h>
#include <cstdint>
#include "macros.h"

namespace vision {
VISION_API int64_t cuda_version() noexcept;

namespace detail {
// Dummy variable to reference a symbol from vision.cpp.
// This ensures that the torchvision library and the ops registration
// initializers are not pruned.
VISION_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace vision

#endif // VISION_H
