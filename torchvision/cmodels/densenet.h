#ifndef DENSENET_H
#define DENSENET_H

#include "visionimpl.h"

namespace torchvision
{
namespace densenetimpl
{
// TODO give modules names in sequential subclasses
class _DenseLayerImpl : public torch::nn::SequentialImpl
{
	double drop_rate;

public:
	_DenseLayerImpl(int64_t num_input_features, int64_t growth_rate,
					int64_t bn_size, double drop_rate);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(_DenseLayer);

class _DenseBlockImpl : public torch::nn::SequentialImpl
{
public:
	_DenseBlockImpl(int64_t num_layers, int64_t num_input_features,
					int64_t bn_size, int64_t growth_rate, double drop_rate);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(_DenseBlock);

class _TransitionImpl : public torch::nn::SequentialImpl
{
public:
	_TransitionImpl(int64_t num_input_features, int64_t num_output_features);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(_Transition);
}  // namespace densenetimpl

class DenseNetImpl : public torch::nn::Module
{
	torch::nn::Sequential features{nullptr};
	torch::nn::Linear classifier{nullptr};

public:
	DenseNetImpl(int64_t growth_rate = 32,
				 std::vector<int64_t> block_config = {6, 12, 24, 16},
				 int64_t num_init_features = 64, int64_t bn_size = 4,
				 int64_t drop_rate = 0, int64_t num_classes = 1000);

	torch::Tensor forward(torch::Tensor x);
};

class DenseNet121Impl : public DenseNetImpl
{
public:
	DenseNet121Impl();
};

class DenseNet169Impl : public DenseNetImpl
{
public:
	DenseNet169Impl();
};

class DenseNet201Impl : public DenseNetImpl
{
public:
	DenseNet201Impl();
};

class DenseNet161Impl : public DenseNetImpl
{
public:
	DenseNet161Impl();
};

TORCH_MODULE(DenseNet121);
TORCH_MODULE(DenseNet169);
TORCH_MODULE(DenseNet201);
TORCH_MODULE(DenseNet161);

}  // namespace torchvision

#endif  // DENSENET_H
