#include "resnet.h"

int torchvision::resnetimpl::BasicBlock::expantion = 1;
int torchvision::resnetimpl::Bottleneck::expantion = 4;

torch::nn::Conv2d torchvision::resnetimpl::conv3x3(int64_t in, int64_t out,
												   int64_t stride)
{
	torch::nn::Conv2dOptions O(in, out, 3);
	O.padding(1).stride(stride).with_bias(false);
	return torch::nn::Conv2d(O);
}

torch::nn::Conv2d torchvision::resnetimpl::conv1x1(int64_t in, int64_t out,
												   int64_t stride)
{
	torch::nn::Conv2dOptions O(in, out, 1);
	O.stride(stride).with_bias(false);
	return torch::nn::Conv2d(O);
}

torchvision::resnetimpl::BasicBlock::BasicBlock(
	int64_t inplanes, int64_t planes, int64_t stride,
	torch::nn::Sequential downsample)
	: stride(stride),
	  downsample(downsample),
	  conv1(conv3x3(inplanes, planes, stride)),
	  conv2(conv3x3(planes, planes)),
	  bn1(planes),
	  bn2(planes)
{
	register_module("conv1", conv1);
	register_module("conv2", conv2);

	register_module("bn1", bn1);
	register_module("bn2", bn2);

	if (!downsample.is_empty()) register_module("downsample", downsample);
}

torch::Tensor torchvision::resnetimpl::BasicBlock::forward(at::Tensor X)
{
	auto identity = X;

	auto out = conv1->forward(X);
	out = bn1->forward(out).relu_();

	out = conv2->forward(out);
	out = bn2->forward(out);

	if (!downsample.is_empty()) identity = downsample->forward(X);

	out += identity;
	return out.relu_();
}

torchvision::resnetimpl::Bottleneck::Bottleneck(
	int64_t inplanes, int64_t planes, int64_t stride,
	torch::nn::Sequential downsample)
	: stride(stride),
	  downsample(downsample),
	  conv1(conv1x1(inplanes, planes)),
	  conv2(conv3x3(planes, planes, stride)),
	  conv3(conv1x1(planes, planes * expantion)),
	  bn1(planes),
	  bn2(planes),
	  bn3(planes * expantion)
{
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("conv3", conv3);

	register_module("bn1", bn1);
	register_module("bn2", bn2);
	register_module("bn3", bn3);

	if (!downsample.is_empty()) register_module("downsample", downsample);
}

torch::Tensor torchvision::resnetimpl::Bottleneck::forward(at::Tensor X)
{
	auto identity = X;

	auto out = conv1->forward(X);
	out = bn1->forward(out).relu_();

	out = conv2->forward(out);
	out = bn2->forward(out).relu_();

	out = conv3->forward(out);
	out = bn3->forward(out);

	if (!downsample.is_empty()) identity = downsample->forward(X);

	out += identity;
	return out.relu_();
}

torchvision::ResNet18Impl::ResNet18Impl(int classes, bool zeroInitResidual)
	: ResNetImpl({2, 2, 2, 2}, classes, zeroInitResidual)
{
}

torchvision::ResNet34Impl::ResNet34Impl(int classes, bool zeroInitResidual)
	: ResNetImpl({3, 4, 6, 3}, classes, zeroInitResidual)
{
}

torchvision::ResNet50Impl::ResNet50Impl(int classes, bool zeroInitResidual)
	: ResNetImpl({3, 4, 6, 3}, classes, zeroInitResidual)
{
}

torchvision::ResNet101Impl::ResNet101Impl(int classes, bool zeroInitResidual)
	: ResNetImpl({3, 4, 23, 3}, classes, zeroInitResidual)
{
}

torchvision::ResNet152Impl::ResNet152Impl(int classes, bool zeroInitResidual)
	: ResNetImpl({3, 8, 36, 3}, classes, zeroInitResidual)
{
}
