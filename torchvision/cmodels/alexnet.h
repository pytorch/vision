#ifndef ALEXNET_H
#define ALEXNET_H

#include "visionimpl.h"

namespace torchvision
{
class AlexNetImpl : public torch::nn::Module
{
	torch::nn::Sequential features, classifier;

public:
	AlexNetImpl(int classes = 1000);

	torch::Tensor forward(torch::Tensor X);
};

TORCH_MODULE(AlexNet);

}  // namespace torchvision

#endif  // ALEXNET_H
