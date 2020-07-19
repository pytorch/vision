#include "ops.h"

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define VISION_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define VISION_INLINE_VARIABLE __declspec(selectany)
#else
#define VISION_INLINE_VARIABLE __attribute__((weak))
#endif
#endif

#ifdef _MSC_VER
#define VISION_NO_OPTIMIZE_STATIC_VARIABLE __declspec(dllexport)
#else
#define VISION_NO_OPTIMIZE_STATIC_VARIABLE
#endif

namespace vision {
namespace impl {
VISION_INLINE_VARIABLE VISION_NO_OPTIMIZE_STATIC_VARIABLE int dummy = RegisterOps();
}
} // namespace vision
