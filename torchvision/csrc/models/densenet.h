#ifndef DENSENET_H
#define DENSENET_H

#include <torch/torch.h>

namespace torchvision
{
// Densenet-BC model class, based on
// "Densely Connected Convolutional Networks"
// <https://arxiv.org/pdf/1608.06993.pdf>

// Args:
//     growth_rate (int) - how many filters to add each layer (`k` in paper)
//     block_config (list of 4 ints) - how many layers in each pooling block
//     num_init_features (int) - the number of filters to learn in the first
//         convolution layer
//     bn_size (int) - multiplicative factor for number of bottle neck layers
//         (i.e. bn_size * k features in the bottleneck layer)
//     drop_rate (float) - dropout rate after each dense layer
//     num_classes (int) - number of classification classes
class DenseNetImpl : public torch::nn::Module
{
	torch::nn::Sequential features{nullptr};
	torch::nn::Linear classifier{nullptr};

public:
	DenseNetImpl(int64_t num_classes = 1000, int64_t growth_rate = 32,
				 std::vector<int64_t> block_config = {6, 12, 24, 16},
				 int64_t num_init_features = 64, int64_t bn_size = 4,
				 double drop_rate = 0);

	torch::Tensor forward(torch::Tensor x);
};

class DenseNet121Impl : public DenseNetImpl
{
public:
	DenseNet121Impl(int64_t num_classes = 1000, int64_t growth_rate = 32,
					std::vector<int64_t> block_config = {6, 12, 24, 16},
					int64_t num_init_features = 64, int64_t bn_size = 4,
					double drop_rate = 0);
};

class DenseNet169Impl : public DenseNetImpl
{
public:
	DenseNet169Impl(int64_t num_classes = 1000, int64_t growth_rate = 32,
					std::vector<int64_t> block_config = {6, 12, 32, 32},
					int64_t num_init_features = 64, int64_t bn_size = 4,
					double drop_rate = 0);
};

class DenseNet201Impl : public DenseNetImpl
{
public:
	DenseNet201Impl(int64_t num_classes = 1000, int64_t growth_rate = 32,
					std::vector<int64_t> block_config = {6, 12, 48, 32},
					int64_t num_init_features = 64, int64_t bn_size = 4,
					double drop_rate = 0);
};

class DenseNet161Impl : public DenseNetImpl
{
public:
	DenseNet161Impl(int64_t num_classes = 1000, int64_t growth_rate = 48,
					std::vector<int64_t> block_config = {6, 12, 36, 24},
					int64_t num_init_features = 96, int64_t bn_size = 4,
					double drop_rate = 0);
};

TORCH_MODULE(DenseNet);
TORCH_MODULE(DenseNet121);
TORCH_MODULE(DenseNet169);
TORCH_MODULE(DenseNet201);
TORCH_MODULE(DenseNet161);

}  // namespace torchvision

#endif  // DENSENET_H
