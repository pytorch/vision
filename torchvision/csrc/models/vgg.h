#ifndef VGG_H
#define VGG_H

#include <torch/torch.h>

namespace torchvision
{
class VGGImpl : public torch::nn::Module
{
	torch::nn::Sequential features{nullptr}, classifier{nullptr};

	void _initialize_weights();

public:
	VGGImpl(torch::nn::Sequential features, int64_t num_classes = 1000,
			bool initialize_weights = true);

	torch::Tensor forward(torch::Tensor x);
};

class VGG11Impl : public VGGImpl
{
public:
	VGG11Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

class VGG13Impl : public VGGImpl
{
public:
	VGG13Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

class VGG16Impl : public VGGImpl
{
public:
	VGG16Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

class VGG19Impl : public VGGImpl
{
public:
	VGG19Impl(int64_t num_classes = 1000, bool initialize_weights = true);
};

class VGG11BNImpl : public VGGImpl
{
public:
	VGG11BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

class VGG13BNImpl : public VGGImpl
{
public:
	VGG13BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

class VGG16BNImpl : public VGGImpl
{
public:
	VGG16BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
};

class VGG19BNImpl : public VGGImpl
{
public:
	VGG19BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
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
