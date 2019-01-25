#ifndef MODELSIMPL_H
#define MODELSIMPL_H

#include <torch/torch.h>

namespace vision
{
namespace models
{
namespace modelsimpl
{
class ReluImpl : public torch::nn::Module
{
	bool inplace;

public:
	ReluImpl(bool inplace = false);

	torch::Tensor forward(torch::Tensor X);
};

struct MaxPool2DOptions
{
	MaxPool2DOptions(torch::ExpandingArray<2> kernel_size)
		: kernel_size_(kernel_size), stride_(kernel_size)
	{
	}

	TORCH_ARG(torch::ExpandingArray<2>, kernel_size);
	TORCH_ARG(torch::ExpandingArray<2>, stride);
	TORCH_ARG(torch::ExpandingArray<2>, padding) = 0;
	TORCH_ARG(torch::ExpandingArray<2>, dilation) = 1;
	TORCH_ARG(bool, ceil_mode) = false;
};

class MaxPool2DImpl : public torch::nn::Module
{
	MaxPool2DOptions options;

public:
	MaxPool2DImpl(torch::ExpandingArray<2> kernel,
				  torch::ExpandingArray<2> stride);

	explicit MaxPool2DImpl(MaxPool2DOptions options);

	torch::Tensor forward(torch::Tensor X);
};

class AdaptiveAvgPool2DImpl : public torch::nn::Module
{
	torch::ExpandingArray<2> output_size;

public:
	AdaptiveAvgPool2DImpl(torch::ExpandingArray<2> output_size);

	torch::Tensor forward(torch::Tensor x);
};

class AvgPool2DImpl : public torch::nn::Module
{
	torch::ExpandingArray<2> kernel_size, stride;

public:
	AvgPool2DImpl(torch::ExpandingArray<2> kernel_size,
				  torch::ExpandingArray<2> stride);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Relu);
TORCH_MODULE(MaxPool2D);
TORCH_MODULE(AvgPool2D);
TORCH_MODULE(AdaptiveAvgPool2D);

}  // namespace modelsimpl
}  // namespace models
}  // namespace vision

#endif  // MODELSIMPL_H
