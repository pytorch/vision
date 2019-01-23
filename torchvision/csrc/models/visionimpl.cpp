#include "visionimpl.h"

namespace torchvision
{
namespace visionimpl
{
ReluImpl::ReluImpl(bool inplace) : torch::nn::Module(), inplace(inplace) {}

torch::Tensor ReluImpl::forward(torch::Tensor X)
{
	if (inplace)
	{
		X.relu_();
		return X;
	}

	return torch::relu(X);
}

MaxPool2DImpl::MaxPool2DImpl(int64_t kernel, int64_t stride, bool ceil_mode,
							 c10::IntList padding)
	: torch::nn::Module(),
	  kernel(kernel),
	  stride(stride),
	  ceil_mode(ceil_mode),
	  padding(padding)
{
}

torch::Tensor MaxPool2DImpl::forward(torch::Tensor X)
{
	return torch::max_pool2d(X, kernel, stride, padding, 1, ceil_mode);
}

AdaptiveAvgPool2DImpl::AdaptiveAvgPool2DImpl(c10::IntList output_size)
	: output_size(output_size)
{
}

torch::Tensor AdaptiveAvgPool2DImpl::forward(torch::Tensor x)
{
	return torch::adaptive_avg_pool2d(x, output_size);
}

AvgPool2DImpl::AvgPool2DImpl(c10::IntList kernel_size, c10::IntList stride)
	: kernel_size(kernel_size), stride(stride)
{
}

torch::Tensor AvgPool2DImpl::forward(torch::Tensor x)
{
	return torch::avg_pool2d(x, kernel_size, stride);
}

}  // namespace visionimpl
}  // namespace torchvision
