#pragma once

#include <ATen/ATen.h>

namespace vision {
namespace ops {

at::Tensor new_empty_tensor(
    const at::Tensor& input,
    const c10::List<int64_t>& shape);

} // namespace ops
} // namespace vision
