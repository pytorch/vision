#pragma once

// All pure C++ headers for the C++ frontend.
#include <torch/all.h>
// Python bindings for the C++ frontend (includes Python.h).
#include <torch/python.h>

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class NewEmptyTensorOp : public torch::autograd::Function<NewEmptyTensorOp> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input,
      c10::List<int64_t> new_shape) {
    ctx->saved_data["shape"] = input.sizes();
    std::vector<int64_t> shape(new_shape.begin(), new_shape.end());
    return {input.new_empty(shape, TensorOptions())};
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    // Use data saved in forward
    auto shape = ctx->saved_data["shape"].toIntList();
    auto out = forward(ctx, grad_output[0], shape);
    return {out[0], at::Tensor()};
  }
};

Tensor new_empty_tensor(const Tensor& input, c10::List<int64_t> shape) {
  return NewEmptyTensorOp::apply(input, shape)[0];
}
