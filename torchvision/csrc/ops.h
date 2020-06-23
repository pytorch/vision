#ifndef TORCHVISION_OPS_H
#define TORCHVISION_OPS_H

#include <c10/macros/Export.h>
#include <cstdint>

namespace vision {
int RegisterOps() noexcept;
}

C10_IMPORT int64_t _cuda_version();

#endif // TORCHVISION_OPS_H
