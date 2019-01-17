#ifndef VGG_H
#define VGG_H

#include "visionimpl.h"

namespace torchvision
{
class VGGImpl : public torch::nn::Module
{
	torch::nn::Sequential features;
	torch::nn::Sequential classifier;

	void initializeWeights();

public:
	VGGImpl(torch::nn::Sequential features, int classes = 1000,
			bool initWeights = true);

	torch::Tensor forward(torch::Tensor X);
};

torch::nn::Sequential makeLayers(const std::vector<int> &cfg,
								 bool batchNorm = false);

class VGG11Impl : public VGGImpl
{
public:
	VGG11Impl(int classes = 1000, bool initWeights = true);
};

class VGG13Impl : public VGGImpl
{
public:
	VGG13Impl(int classes = 1000, bool initWeights = true);
};

class VGG16Impl : public VGGImpl
{
public:
	VGG16Impl(int classes = 1000, bool initWeights = true);
};

class VGG19Impl : public VGGImpl
{
public:
	VGG19Impl(int classes = 1000, bool initWeights = true);
};

class VGG11BNImpl : public VGGImpl
{
public:
	VGG11BNImpl(int classes = 1000, bool initWeights = true);
};

class VGG13BNImpl : public VGGImpl
{
public:
	VGG13BNImpl(int classes = 1000, bool initWeights = true);
};

class VGG16BNImpl : public VGGImpl
{
public:
	VGG16BNImpl(int classes = 1000, bool initWeights = true);
};

class VGG19BNImpl : public VGGImpl
{
public:
	VGG19BNImpl(int classes = 1000, bool initWeights = true);
};

TORCH_MODULE(VGG11);
TORCH_MODULE(VGG13);
TORCH_MODULE(VGG16);
TORCH_MODULE(VGG19);

TORCH_MODULE(VGG11BN);
TORCH_MODULE(VGG13BN);
TORCH_MODULE(VGG16BN);
TORCH_MODULE(VGG19BN);

}  // namespace torchvision

#endif  // VGG_H
