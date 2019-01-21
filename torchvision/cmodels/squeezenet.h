#ifndef SQUEEZENET_H
#define SQUEEZENET_H

#include "visionimpl.h"

namespace torchvision
{
namespace squeezenetimpl
{
class Fire : public torch::nn::Module
{
	torch::nn::Conv2d squeeze, expand1x1, expand3x3;

public:
	Fire(int64_t inplanes, int64_t squeeze_planes, int64_t expand1x1_planes,
		 int64_t expand3x3_planes);

	torch::Tensor forward(torch::Tensor x);
};

}  // namespace squeezenetimpl

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
