#include "alexnet.h"

torchvision::AlexNetImpl::AlexNetImpl(int classes)
{
    // clang-format off
    features = torch::nn::Sequential(
				visionimpl::Conv(3, 64, 11, 2, 4),
				visionimpl::Relu(true),
				visionimpl::MaxPool(3, 2),
				visionimpl::Conv(64, 192, 5, 2),
				visionimpl::Relu(true),
				visionimpl::MaxPool(3, 2),
				visionimpl::Conv(192, 384, 3, 1),
				visionimpl::Relu(true),
				visionimpl::Conv(384, 256, 3, 1),
				visionimpl::Relu(true),
				visionimpl::Conv(256, 256, 3, 1),
				visionimpl::Relu(true),
				visionimpl::MaxPool(3, 2));

    classifier = torch::nn::Sequential(
                torch::nn::Dropout(),
                torch::nn::Linear(256 * 6 * 6, 4096),
				visionimpl::Relu(true),
                torch::nn::Dropout(),
                torch::nn::Linear(4096, 4096),
				visionimpl::Relu(true),
                torch::nn::Linear(4096, classes));
    // clang-format on

    register_module("features", features);
    register_module("clasifier", classifier);
}

torch::Tensor torchvision::AlexNetImpl::forward(at::Tensor X)
{
    X = features->forward(X);
    X = X.view({X.size(0), 256 * 6 * 6});
    X = classifier->forward(X);

    return X;
}
