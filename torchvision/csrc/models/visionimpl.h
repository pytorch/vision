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
	inline ReluImpl(bool inplace = false)
		: torch::nn::Module(), inplace(inplace)
	{
	}

	inline torch::Tensor forward(torch::Tensor X)
	{
		if (inplace)
        {
			X.relu_();
			return X;
		}

		return torch::relu(X);
	}
};

class MaxPool2DImpl : public torch::nn::Module
{
	int64_t kernel;
	int64_t stride;
	bool ceil_mode;
	torch::IntList padding;

public:
	inline MaxPool2DImpl(int64_t kernel, int64_t stride, bool ceil_mode = false,
						 torch::IntList padding = torch::IntList({0}))
		: torch::nn::Module(),
		  kernel(kernel),
		  stride(stride),
		  ceil_mode(ceil_mode),
		  padding(padding)
	{
	}

	inline torch::Tensor forward(torch::Tensor X)
	{
		return torch::max_pool2d(X, kernel, stride, padding, 1, ceil_mode);
	}
};

class AdaptiveAvgPool2DImpl : public torch::nn::Module
{
	torch::IntList output_size;

public:
	AdaptiveAvgPool2DImpl(torch::IntList output_size) : output_size(output_size)
	{
	}

	inline torch::Tensor forward(torch::Tensor x)
	{
		return torch::adaptive_avg_pool2d(x, output_size);
	}
};

class AvgPool2DImpl : public torch::nn::Module
{
	torch::IntList kernel_size, stride;

public:
	AvgPool2DImpl(torch::IntList kernel_size, torch::IntList stride)
		: kernel_size(kernel_size), stride(stride)
	{
	}

	inline torch::Tensor forward(torch::Tensor x)
	{
		return torch::avg_pool2d(x, kernel_size, stride);
	}
};

TORCH_MODULE(Relu);
TORCH_MODULE(MaxPool2D);
TORCH_MODULE(AvgPool2D);
TORCH_MODULE(AdaptiveAvgPool2D);

}  // namespace visionimpl
}  // namespace torchvision

#endif  // VISIONIMPL_H
