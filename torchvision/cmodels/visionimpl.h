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

public:
	inline MaxPoolImpl(int64_t kernel, int64_t stride)
		: torch::nn::Module(), kernel(kernel), stride(stride)
	{
	}

	inline torch::Tensor forward(torch::Tensor X)
	{
		return torch::max_pool2d(X, kernel, stride);
	}
};

TORCH_MODULE(Relu);
TORCH_MODULE(MaxPool);

}  // namespace visionimpl
}  // namespace torchvision

#endif  // VISIONIMPL_H
