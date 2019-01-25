#include "visionimpl.h"

namespace vision
{
namespace visionimpl
{
ReluImpl::ReluImpl(bool inplace) : inplace(inplace) {}

torch::Tensor ReluImpl::forward(torch::Tensor X)
{
	if (inplace)
	{
		X.relu_();
		return X;
	}

	return torch::relu(X);
}

MaxPool2DImpl::MaxPool2DImpl(torch::ExpandingArray<2> kernel,
							 torch::ExpandingArray<2> stride)
	: options(MaxPool2DOptions(kernel).stride(stride))
{
}

MaxPool2DImpl::MaxPool2DImpl(MaxPool2DOptions options) : options(options) {}

torch::Tensor MaxPool2DImpl::forward(torch::Tensor X)
{
	return torch::max_pool2d(X, options.kernel_size(), options.stride(),
							 options.padding(), options.dilation(),
							 options.ceil_mode());
}

AdaptiveAvgPool2DImpl::AdaptiveAvgPool2DImpl(
	torch::ExpandingArray<2> output_size)
	: output_size(output_size)
{
}

torch::Tensor AdaptiveAvgPool2DImpl::forward(torch::Tensor x)
{
	return torch::adaptive_avg_pool2d(x, output_size);
}

AvgPool2DImpl::AvgPool2DImpl(torch::ExpandingArray<2> kernel_size,
							 torch::ExpandingArray<2> stride)
	: kernel_size(kernel_size), stride(stride)
{
}

torch::Tensor AvgPool2DImpl::forward(torch::Tensor x)
{
	return torch::avg_pool2d(x, kernel_size, stride);
}

}  // namespace visionimpl
}  // namespace torchvision
