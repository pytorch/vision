#include "alexnet.h"

#include "modelsimpl.h"

namespace vision {
namespace models {
AlexNetImpl::AlexNetImpl(int64_t num_classes) {
  // clang-format off
    features = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)),
				modelsimpl::Relu(true),
				modelsimpl::MaxPool2D(3, 2),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2)),
				modelsimpl::Relu(true),
				modelsimpl::MaxPool2D(3, 2),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
				modelsimpl::Relu(true),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
				modelsimpl::Relu(true),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
				modelsimpl::Relu(true),
				modelsimpl::MaxPool2D(3, 2));

    classifier = torch::nn::Sequential(
				torch::nn::Dropout(),
				torch::nn::Linear(256 * 6 * 6, 4096),
				modelsimpl::Relu(true),
				torch::nn::Dropout(),
				torch::nn::Linear(4096, 4096),
				modelsimpl::Relu(true),
				torch::nn::Linear(4096, num_classes));
  // clang-format on

  register_module("features", features);
  register_module("clasifier", classifier);
}

torch::Tensor AlexNetImpl::forward(torch::Tensor x) {
  x = features->forward(x);
  x = x.view({x.size(0), 256 * 6 * 6});
  x = classifier->forward(x);

  return x;
}

} // namespace models
} // namespace vision
