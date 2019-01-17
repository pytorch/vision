#ifndef VISIONIMPL_H
#define VISIONIMPL_H

#include <torch/torch.h>

namespace torchvision
{
namespace visionimpl
{
class Conv : public torch::nn::Conv2d
{
public:
	inline Conv(int64_t input, int64_t output, int64_t kernel,
				int64_t padding = 0, int64_t stride = 1)
		: Conv2d(torch::nn::Conv2dOptions(input, output, kernel)
					 .padding(padding)
					 .stride(stride))
	{
	}
};

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

class MaxPoolImpl : public torch::nn::Module
{
	int64_t kernel;
	int64_t stride;
	bool ceil_mode;

public:
	inline MaxPoolImpl(int64_t kernel, int64_t stride, bool ceil_mode = false)
		: torch::nn::Module(),
		  kernel(kernel),
		  stride(stride),
		  ceil_mode(ceil_mode)
	{
	}

	inline torch::Tensor forward(torch::Tensor X)
	{
		return torch::max_pool2d(X, kernel, stride, 0, 1, ceil_mode);
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

TORCH_MODULE(Relu);
TORCH_MODULE(MaxPool);
TORCH_MODULE(AdaptiveAvgPool2D);

}  // namespace visionimpl
}  // namespace torchvision

#endif  // VISIONIMPL_H
