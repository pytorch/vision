#ifndef VGG_H
#define VGG_H

#include "visionimpl.h"

namespace torchvision
{
class VGGImpl : public torch::nn::Module
{
	torch::nn::Sequential features;
	torch::nn::Sequential classifier;

	void _initialize_weights();

public:
	VGGImpl(torch::nn::Sequential features, int num_classes = 1000,
			bool initialize_weights = true);

	torch::Tensor forward(torch::Tensor x);
};

torch::nn::Sequential makeLayers(const std::vector<int> &cfg,
								 bool batch_norm = false);

class VGG11Impl : public VGGImpl
{
public:
	VGG11Impl(int num_classes = 1000, bool initialize_weights = true);
};

class VGG13Impl : public VGGImpl
{
public:
	VGG13Impl(int num_classes = 1000, bool initialize_weights = true);
};

class VGG16Impl : public VGGImpl
{
public:
	VGG16Impl(int num_classes = 1000, bool initialize_weights = true);
};

class VGG19Impl : public VGGImpl
{
public:
	VGG19Impl(int num_classes = 1000, bool initialize_weights = true);
};

class VGG11BNImpl : public VGGImpl
{
public:
	VGG11BNImpl(int num_classes = 1000, bool initialize_weights = true);
};

class VGG13BNImpl : public VGGImpl
{
public:
	VGG13BNImpl(int num_classes = 1000, bool initialize_weights = true);
};

class VGG16BNImpl : public VGGImpl
{
public:
	VGG16BNImpl(int num_classes = 1000, bool initialize_weights = true);
};

class VGG19BNImpl : public VGGImpl
{
public:
	VGG19BNImpl(int num_classes = 1000, bool initialize_weights = true);
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
