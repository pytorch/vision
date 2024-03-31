#pragma once

#include <cstdint>
#include "macros.h"

namespace vision {
VISION_API int64_t cuda_version();

namespace detail {
extern "C" VISION_INLINE_VARIABLE auto _register_ops = &cuda_version;
#ifdef HINT_MSVC_LINKER_INCLUDE_SYMBOL
#pragma comment(linker, "/include:_register_ops")
#endif

} // namespace detail
} // namespace vision
