#ifndef VISION_H
#define VISION_H

#include <torchvision/models/models.h>
#include "macros.h"

namespace vision {
VISION_API int RegisterOps() noexcept;

namespace detail {
VISION_INLINE_VARIABLE int dummy = RegisterOps();
}
} // namespace vision

#endif // VISION_H
