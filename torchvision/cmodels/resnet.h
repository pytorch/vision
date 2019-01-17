#ifndef RESNET_H
#define RESNET_H

#include "visionimpl.h"

namespace torchvision
{
template <typename Block>
class ResNetImpl;

namespace resnetimpl
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

	torch::Tensor forward(torch::Tensor X);
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
}  // namespace resnetimpl

template <typename Block>
class ResNetImpl : public torch::nn::Module
{
	int64_t inplanes;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm bn1;
	torch::nn::Linear fc;
	torch::nn::Sequential layer1, layer2, layer3, layer4;

	torch::nn::Sequential makeLayer(int64_t planes, int64_t blocks,
									int64_t stride = 1);

public:
	ResNetImpl(const std::vector<int> &layers, int classes = 1000,
			   bool zeroInitResidual = false);

	torch::Tensor forward(torch::Tensor X);
};

template <typename Block>
torch::nn::Sequential ResNetImpl<Block>::makeLayer(int64_t planes,
												   int64_t blocks,
												   int64_t stride)
{
	torch::nn::Sequential downsample = nullptr;
	if (stride != 1 || inplanes != planes * Block::expantion)
	{
		downsample = torch::nn::Sequential(
			resnetimpl::conv1x1(inplanes, planes * Block::expantion, stride),
			torch::nn::BatchNorm(planes * Block::expantion));
	}

	torch::nn::Sequential layers;
	layers->push_back(Block(inplanes, planes, stride, downsample));

	inplanes = planes * Block::expantion;

	for (int i = 1; i < blocks; ++i) layers->push_back(Block(inplanes, planes));

	return layers;
}

template <typename Block>
ResNetImpl<Block>::ResNetImpl(const std::vector<int> &layers, int classes,
							  bool zeroInitResidual)
	: inplanes(64),
	  conv1(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).with_bias(
		  false)),
	  bn1(64),
	  layer1(makeLayer(64, layers[0])),
	  layer2(makeLayer(128, layers[1], 2)),
	  layer3(makeLayer(256, layers[2], 2)),
	  layer4(makeLayer(512, layers[3], 2)),
	  fc(512 * Block::expantion, classes)
{
	if (zeroInitResidual)
		for (auto &module : modules(false))
		{
			if (resnetimpl::Bottleneck *M =
					dynamic_cast<resnetimpl::Bottleneck *>(module.get()))
				torch::nn::init::constant_(M->bn3->weight, 0);
			else if (resnetimpl::BasicBlock *M =
						 dynamic_cast<resnetimpl::BasicBlock *>(module.get()))
				torch::nn::init::constant_(M->bn2->weight, 0);
		}

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("fc", fc);

	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
}

template <typename Block>
torch::Tensor ResNetImpl<Block>::forward(at::Tensor X)
{
	X = conv1->forward(X);
	X = bn1->forward(X).relu_();
	X = torch::max_pool2d(X, 3, 2, 1);

	X = layer1->forward(X);
	X = layer2->forward(X);
	X = layer3->forward(X);
	X = layer4->forward(X);

	X = torch::adaptive_avg_pool2d(X, {1, 1});
	X = X.view({X.size(0), -1});
	X = fc->forward(X);

	return X;
}

class ResNet18Impl : public ResNetImpl<resnetimpl::BasicBlock>
{
public:
	ResNet18Impl(int classes = 1000, bool zeroInitResidual = false);
};

class ResNet34Impl : public ResNetImpl<resnetimpl::BasicBlock>
{
public:
	ResNet34Impl(int classes = 1000, bool zeroInitResidual = false);
};

class ResNet50Impl : public ResNetImpl<resnetimpl::Bottleneck>
{
public:
	ResNet50Impl(int classes = 1000, bool zeroInitResidual = false);
};

class ResNet101Impl : public ResNetImpl<resnetimpl::Bottleneck>
{
public:
	ResNet101Impl(int classes = 1000, bool zeroInitResidual = false);
};

class ResNet152Impl : public ResNetImpl<resnetimpl::Bottleneck>
{
public:
	ResNet152Impl(int classes = 1000, bool zeroInitResidual = false);
};

TORCH_MODULE(ResNet18);
TORCH_MODULE(ResNet34);
TORCH_MODULE(ResNet50);
TORCH_MODULE(ResNet101);
TORCH_MODULE(ResNet152);

}  // namespace torchvision

#endif  // RESNET_H
