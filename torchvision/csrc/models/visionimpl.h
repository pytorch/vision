#ifndef VISIONIMPL_H
#define VISIONIMPL_H

#include <torch/torch.h>

namespace torchvision
{
namespace visionimpl
{
class ReluImpl : public torch::nn::Module
{
	bool inplace;

public:
	ReluImpl(bool inplace = false);

	torch::Tensor forward(torch::Tensor X);
};

class MaxPool2DImpl : public torch::nn::Module
{
	int64_t kernel;
	int64_t stride;
	bool ceil_mode;
	torch::IntList padding;

public:
	MaxPool2DImpl(int64_t kernel, int64_t stride, bool ceil_mode = false,
				  torch::IntList padding = torch::IntList({0}));

	torch::Tensor forward(torch::Tensor X);
};

class AdaptiveAvgPool2DImpl : public torch::nn::Module
{
	torch::IntList output_size;

public:
	AdaptiveAvgPool2DImpl(torch::IntList output_size);

	torch::Tensor forward(torch::Tensor x);
};

class AvgPool2DImpl : public torch::nn::Module
{
	torch::IntList kernel_size, stride;

public:
	AvgPool2DImpl(torch::IntList kernel_size, torch::IntList stride);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Relu);
TORCH_MODULE(MaxPool2D);
TORCH_MODULE(AvgPool2D);
TORCH_MODULE(AdaptiveAvgPool2D);

}  // namespace visionimpl
}  // namespace torchvision

#endif  // VISIONIMPL_H
