#pragma once

#include "new_empty_tensor_op.h"
#include <torch/extension.h>

namespace vision {
namespace ops {

namespace {

class NewEmptyTensorOp : public torch::autograd::Function<NewEmptyTensorOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const c10::List<int64_t>& new_shape) {
    ctx->saved_data["shape"] = input.sizes();
    std::vector<int64_t> shape(new_shape.begin(), new_shape.end());
    return {input.new_empty(shape, at::TensorOptions())};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto shape = ctx->saved_data["shape"].toIntList();
    auto out = forward(ctx, grad_output[0], shape);
    return {out[0], at::Tensor()};
  }
};

} // namespace

at::Tensor new_empty_tensor(
    const at::Tensor& input,
    const c10::List<int64_t>& shape) {
  return NewEmptyTensorOp::apply(input, shape)[0];
}

} // namespace ops
} // namespace vision
