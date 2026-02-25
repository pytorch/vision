#pragma once

#include "../StableABICompat.h"
#include "../macros.h"

namespace vision {
namespace ops {

// Note: With stable ABI, nms is called directly via torch.ops.torchvision.nms
// This header is kept for backwards compatibility but the C++ API is deprecated.

} // namespace ops
} // namespace vision
