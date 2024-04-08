#include <iostream>
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include "../macros.h"
#include <torch/library.h>
#include <torch/types.h>

torch::Tensor my_custom_op(torch::Tensor x){
  std::cout << "running my_custom_op!" << std::endl;
  std::cout << x << std::endl;
  auto x_data = x.data_ptr<float>();
  std::cout << "done accessing data_ptr!" << std::endl;
  return x.clone();
}

static auto registry =
  torch::RegisterOperators().op("torchvision::my_custom_op1", &my_custom_op);

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def("my_custom_op2(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
    "my_custom_op2", &my_custom_op
  );
}
