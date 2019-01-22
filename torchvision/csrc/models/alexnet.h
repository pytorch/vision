#ifndef ALEXNET_H
#define ALEXNET_H

#include <torch/torch.h>

namespace torchvision
{
class AlexNetImpl : public torch::nn::Module
{
	torch::nn::Sequential features{nullptr}, classifier{nullptr};

public:
	AlexNetImpl(int64_t num_classes = 1000);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(AlexNet);

}  // namespace torchvision

#endif  // ALEXNET_H
