#include "densenet.h"

#include "visionimpl.h"

namespace torchvision
{
// TODO give modules names in sequential subclasses
class _DenseLayerImpl : public torch::nn::SequentialImpl
{
	double drop_rate;

public:
	_DenseLayerImpl(int64_t num_input_features, int64_t growth_rate,
					int64_t bn_size, double drop_rate)
		: drop_rate(drop_rate)
	{
		push_back(torch::nn::BatchNorm(num_input_features));
		push_back(visionimpl::Relu(true));
		push_back(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(num_input_features,
													   bn_size * growth_rate, 1)
								  .stride(1)
								  .with_bias(false)));
		push_back(torch::nn::BatchNorm(bn_size * growth_rate));
		push_back(visionimpl::Relu(true));
		push_back(torch::nn::Conv2d(
			torch::nn::Conv2dOptions(bn_size * growth_rate, growth_rate, 3)
				.stride(1)
				.padding(1)
				.with_bias(false)));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		auto new_features = torch::nn::SequentialImpl::forward(x);
		if (drop_rate > 0)
			new_features =
				torch::dropout(new_features, drop_rate, this->is_training());
		return torch::cat({x, new_features}, 1);
	}
};

TORCH_MODULE(_DenseLayer);

class _DenseBlockImpl : public torch::nn::SequentialImpl
{
public:
	_DenseBlockImpl(int64_t num_layers, int64_t num_input_features,
					int64_t bn_size, int64_t growth_rate, double drop_rate)
	{
		for (int64_t i = 0; i < num_layers; ++i)
		{
			auto layer = _DenseLayer(num_input_features + i * growth_rate,
									 growth_rate, bn_size, drop_rate);
			push_back(layer);
		}
	}

	torch::Tensor forward(torch::Tensor x)
	{
		return torch::nn::SequentialImpl::forward(x);
	}
};

TORCH_MODULE(_DenseBlock);

class _TransitionImpl : public torch::nn::SequentialImpl
{
public:
	_TransitionImpl(int64_t num_input_features, int64_t num_output_features)
	{
		push_back(torch::nn::BatchNorm(num_input_features));
		push_back(visionimpl::Relu(true));
		push_back(torch::nn::Conv2d(
			torch::nn::Conv2dOptions(num_input_features, num_output_features, 1)
				.stride(1)
				.with_bias(false)));
		push_back(
			visionimpl::AvgPool2D(torch::IntList({2}), torch::IntList({2})));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		return torch::nn::SequentialImpl::forward(x);
	}
};

TORCH_MODULE(_Transition);

DenseNetImpl::DenseNetImpl(int64_t num_classes, int64_t growth_rate,
						   std::vector<int64_t> block_config,
						   int64_t num_init_features, int64_t bn_size,
						   double drop_rate)
{
	// First convolution
	features = torch::nn::Sequential(
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, num_init_features, 7)
							  .stride(2)
							  .padding(3)
							  .with_bias(false)),
		torch::nn::BatchNorm(num_init_features), visionimpl::Relu(true),
		visionimpl::MaxPool2D(3, 2, false, torch::IntList({1})));

	// Each denseblock
	auto num_features = num_init_features;
	for (size_t i = 0; i < block_config.size(); ++i)
	{
		auto num_layers = block_config[i];
		auto block = _DenseBlock(num_layers, num_features, bn_size, growth_rate,
								 drop_rate);
		features->push_back(block);
		num_features = num_features + num_layers * growth_rate;

		if (i != block_config.size() - 1)
		{
			auto trans = _Transition(num_features, num_features / 2);
			features->push_back(trans);
			num_features = num_features / 2;
		}
	}

	// Final batch norm
	features->push_back(torch::nn::BatchNorm(num_features));
	// Linear layer
	classifier = torch::nn::Linear(num_features, num_classes);

	register_module("features", features);
	register_module("classifier", classifier);

	// Official init from torch repo.
	for (auto &module : modules(false))
	{
		if (torch::nn::Conv2dImpl *M =
				dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
			torch::nn::init::xavier_normal_(M->weight);
		else if (torch::nn::BatchNormImpl *M =
					 dynamic_cast<torch::nn::BatchNormImpl *>(module.get()))
		{
			torch::nn::init::constant_(M->weight, 1);
			torch::nn::init::constant_(M->bias, 0);
		}
		else if (torch::nn::LinearImpl *M =
					 dynamic_cast<torch::nn::LinearImpl *>(module.get()))
			torch::nn::init::normal_(M->bias, 0);
	}
}

torch::Tensor DenseNetImpl::forward(torch::Tensor x)
{
	auto features = this->features->forward(x);
	auto out = torch::relu_(features);
	out = torch::adaptive_avg_pool2d(out, {1, 1}).view({features.size(0), -1});
	out = this->classifier->forward(out);
	return out;
}

DenseNet121Impl::DenseNet121Impl(int64_t num_classes, int64_t growth_rate,
								 std::vector<int64_t> block_config,
								 int64_t num_init_features, int64_t bn_size,
								 double drop_rate)
	: DenseNetImpl(num_classes, growth_rate, block_config, num_init_features,
				   bn_size, drop_rate)
{
}

DenseNet169Impl::DenseNet169Impl(int64_t num_classes, int64_t growth_rate,
								 std::vector<int64_t> block_config,
								 int64_t num_init_features, int64_t bn_size,
								 double drop_rate)
	: DenseNetImpl(num_classes, growth_rate, block_config, num_init_features,
				   bn_size, drop_rate)
{
}

DenseNet201Impl::DenseNet201Impl(int64_t num_classes, int64_t growth_rate,
								 std::vector<int64_t> block_config,
								 int64_t num_init_features, int64_t bn_size,
								 double drop_rate)
	: DenseNetImpl(num_classes, growth_rate, block_config, num_init_features,
				   bn_size, drop_rate)
{
}

DenseNet161Impl::DenseNet161Impl(int64_t num_classes, int64_t growth_rate,
								 std::vector<int64_t> block_config,
								 int64_t num_init_features, int64_t bn_size,
								 double drop_rate)
	: DenseNetImpl(num_classes, growth_rate, block_config, num_init_features,
				   bn_size, drop_rate)
{
}

}  // namespace torchvision
