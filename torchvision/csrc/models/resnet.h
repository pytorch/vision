#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>

namespace torchvision
{
template <typename Block>
class ResNetImpl;

namespace _resnetimpl
{
torch::nn::Conv2d conv3x3(int64_t in, int64_t out, int64_t stride = 1);
torch::nn::Conv2d conv1x1(int64_t in, int64_t out, int64_t stride = 1);

class BasicBlock : public torch::nn::Module
{
	template <typename Block>
	friend class torchvision::ResNetImpl;

	int64_t stride;
	torch::nn::Sequential downsample;

	torch::nn::Conv2d conv1, conv2;
	torch::nn::BatchNorm bn1, bn2;

public:
	static int expantion;

	BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
			   torch::nn::Sequential downsample = nullptr);

	torch::Tensor forward(torch::Tensor x);
};

class Bottleneck : public torch::nn::Module
{
	template <typename Block>
	friend class torchvision::ResNetImpl;

	int64_t stride;
	torch::nn::Sequential downsample;

	torch::nn::Conv2d conv1, conv2, conv3;
	torch::nn::BatchNorm bn1, bn2, bn3;

public:
	static int expantion;

	Bottleneck(int64_t inplanes, int64_t planes, int64_t stride = 1,
			   torch::nn::Sequential downsample = nullptr);

	torch::Tensor forward(torch::Tensor X);
};
}  // namespace _resnetimpl

template <typename Block>
class ResNetImpl : public torch::nn::Module
{
	int64_t inplanes;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm bn1;
	torch::nn::Linear fc;
	torch::nn::Sequential layer1, layer2, layer3, layer4;

	torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks,
									  int64_t stride = 1);

public:
	ResNetImpl(const std::vector<int> &layers, int64_t num_classes = 1000,
			   bool zero_init_residual = false);

	torch::Tensor forward(torch::Tensor X);
};

template <typename Block>
torch::nn::Sequential ResNetImpl<Block>::_make_layer(int64_t planes,
													 int64_t blocks,
													 int64_t stride)
{
	// TODO Blocks that are created here are not shared_ptr. see if that is a
	// problem (the layers in blocks are used in the resnet constructor).

	torch::nn::Sequential downsample = nullptr;
	if (stride != 1 || inplanes != planes * Block::expantion)
	{
		downsample = torch::nn::Sequential(
			_resnetimpl::conv1x1(inplanes, planes * Block::expantion, stride),
			torch::nn::BatchNorm(planes * Block::expantion));
	}

	torch::nn::Sequential layers;
	layers->push_back(Block(inplanes, planes, stride, downsample));

	inplanes = planes * Block::expantion;

	for (int i = 1; i < blocks; ++i) layers->push_back(Block(inplanes, planes));

	return layers;
}

template <typename Block>
ResNetImpl<Block>::ResNetImpl(const std::vector<int> &layers,
							  int64_t num_classes, bool zero_init_residual)
	: inplanes(64),
	  conv1(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).with_bias(
		  false)),
	  bn1(64),
	  layer1(_make_layer(64, layers[0])),
	  layer2(_make_layer(128, layers[1], 2)),
	  layer3(_make_layer(256, layers[2], 2)),
	  layer4(_make_layer(512, layers[3], 2)),
	  fc(512 * Block::expantion, num_classes)
{
	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("fc", fc);

	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);

	for (auto &module : modules(false))
	{
		if (torch::nn::Conv2dImpl *M =
				dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
		{
			torch::nn::init::xavier_normal_(M->weight);
		}
		else if (torch::nn::BatchNormImpl *M =
					 dynamic_cast<torch::nn::BatchNormImpl *>(module.get()))
		{
			torch::nn::init::constant_(M->weight, 1);
			torch::nn::init::constant_(M->bias, 0);
		}
	}

	if (zero_init_residual)
		for (auto &module : modules(false))
		{
			if (_resnetimpl::Bottleneck *M =
					dynamic_cast<_resnetimpl::Bottleneck *>(module.get()))
				torch::nn::init::constant_(M->bn3->weight, 0);
			else if (_resnetimpl::BasicBlock *M =
						 dynamic_cast<_resnetimpl::BasicBlock *>(module.get()))
				torch::nn::init::constant_(M->bn2->weight, 0);
		}
}

template <typename Block>
torch::Tensor ResNetImpl<Block>::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = bn1->forward(x).relu_();
	x = torch::max_pool2d(x, 3, 2, 1);

	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);

	x = torch::adaptive_avg_pool2d(x, {1, 1});
	x = x.view({x.size(0), -1});
	x = fc->forward(x);

	return x;
}

class ResNet18Impl : public ResNetImpl<_resnetimpl::BasicBlock>
{
public:
	ResNet18Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

class ResNet34Impl : public ResNetImpl<_resnetimpl::BasicBlock>
{
public:
	ResNet34Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

class ResNet50Impl : public ResNetImpl<_resnetimpl::Bottleneck>
{
public:
	ResNet50Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

class ResNet101Impl : public ResNetImpl<_resnetimpl::Bottleneck>
{
public:
	ResNet101Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

class ResNet152Impl : public ResNetImpl<_resnetimpl::Bottleneck>
{
public:
	ResNet152Impl(int64_t num_classes = 1000, bool zero_init_residual = false);
};

TORCH_MODULE(ResNet18);
TORCH_MODULE(ResNet34);
TORCH_MODULE(ResNet50);
TORCH_MODULE(ResNet101);
TORCH_MODULE(ResNet152);

}  // namespace torchvision

#endif  // RESNET_H
