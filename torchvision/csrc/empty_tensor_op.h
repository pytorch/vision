#pragma once

// All pure C++ headers for the C++ frontend.
#include <torch/all.h>
// Python bindings for the C++ frontend (includes Python.h).
#include <torch/python.h>

class NewEmptyTensorOp : public torch::autograd::Function<NewEmptyTensorOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::Variable input,
      c10::List<int64_t> new_shape) {
    ctx->saved_data["shape"] = input.sizes();
    std::vector<int64_t> shape(new_shape.begin(), new_shape.end());
    return {input.new_empty(shape, at::TensorOptions())};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    // Use data saved in forward
    auto shape = ctx->saved_data["shape"].toIntList();
    auto out = forward(ctx, grad_output[0], shape);
    return {out[0], at::Tensor()};
  }
};

at::Tensor new_empty_tensor(const at::Tensor& input, c10::List<int64_t> shape) {
  return NewEmptyTensorOp::apply(input, shape)[0];
}
