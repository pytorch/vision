#ifndef VISION_H
#define VISION_H

#include <c10/macros/Macros.h>
#include <torchvision/models/models.h>

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define VISION_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define VISION_INLINE_VARIABLE __declspec(selectany)
#else
#define VISION_INLINE_VARIABLE __attribute__((weak))
#endif
#endif

#ifdef VISION_BUILD_LIB
#define VISION_API C10_EXPORT
#else
#define VISION_API C10_IMPORT
#endif

namespace vision {
VISION_API int RegisterOps() noexcept;

namespace detail {
VISION_INLINE_VARIABLE int dummy = RegisterOps();
}
} // namespace vision

#endif // VISION_H
