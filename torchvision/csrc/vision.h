#ifndef VISION_H
#define VISION_H

#include <torchvision/ops.h>
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

namespace vision {
namespace impl {
VISION_INLINE_VARIABLE int dummy = RegisterOps();
}
}

#endif // VISION_H
