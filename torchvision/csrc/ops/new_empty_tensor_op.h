#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor new_empty_tensor(
    const at::Tensor& input,
    const c10::List<int64_t>& shape);

} // namespace ops
} // namespace vision
