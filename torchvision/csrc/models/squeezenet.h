#ifndef SQUEEZENET_H
#define SQUEEZENET_H

#include <torch/torch.h>

namespace torchvision
{
class SqueezeNetImpl : public torch::nn::Module
{
	int64_t num_classes;
	torch::nn::Sequential features{nullptr}, classifier{nullptr};

public:
	SqueezeNetImpl(double version = 1.0, int64_t num_classes = 1000);

	torch::Tensor forward(torch::Tensor x);
};

class SqueezeNet1_0Impl : public SqueezeNetImpl
{
public:
	SqueezeNet1_0Impl(int64_t num_classes = 1000);
};

class SqueezeNet1_1Impl : public SqueezeNetImpl
{
public:
	SqueezeNet1_1Impl(int64_t num_classes = 1000);
};

TORCH_MODULE(SqueezeNet1_0);
TORCH_MODULE(SqueezeNet1_1);

}  // namespace torchvision

#endif  // SQUEEZENET_H
