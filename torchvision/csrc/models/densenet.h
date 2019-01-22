#ifndef DENSENET_H
#define DENSENET_H

#include <torch/torch.h>

namespace torchvision
{
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
	DenseNet121Impl(int64_t num_classes = 1000);
};

class DenseNet169Impl : public DenseNetImpl
{
public:
	DenseNet169Impl(int64_t num_classes = 1000);
};

class DenseNet201Impl : public DenseNetImpl
{
public:
	DenseNet201Impl(int64_t num_classes = 1000);
};

class DenseNet161Impl : public DenseNetImpl
{
public:
	DenseNet161Impl(int64_t num_classes = 1000);
};

TORCH_MODULE(DenseNet121);
TORCH_MODULE(DenseNet169);
TORCH_MODULE(DenseNet201);
TORCH_MODULE(DenseNet161);

}  // namespace torchvision

#endif  // DENSENET_H
