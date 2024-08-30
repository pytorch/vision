#pragma once

#include <cstdint>
#include "macros.h"

namespace vision {
VISION_API int64_t cuda_version();

namespace detail {
extern "C" inline auto _register_ops = &cuda_version;
} // namespace detail
} // namespace vision
