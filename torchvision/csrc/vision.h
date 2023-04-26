#pragma once

#include <cstdint>
#include "macros.h"

namespace vision {
VISION_API int64_t cuda_version();
VISION_API bool _has_x86_avx2();

namespace detail {
extern "C" VISION_INLINE_VARIABLE auto _register_op1 = &cuda_version;
extern "C" VISION_INLINE_VARIABLE auto _register_op2 = &_has_x86_avx2;
#ifdef HINT_MSVC_LINKER_INCLUDE_SYMBOL
#pragma comment(linker, "/include:_register_op1")
#pragma comment(linker, "/include:_register_op2")
#endif

} // namespace detail
} // namespace vision
