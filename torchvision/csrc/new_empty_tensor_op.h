#pragma once

#include <ATen/ATen.h>

at::Tensor new_empty_tensor(
    const at::Tensor& input,
    const c10::List<int64_t>& shape);
