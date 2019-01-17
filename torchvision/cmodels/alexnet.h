#ifndef ALEXNET_H
#define ALEXNET_H

#include "visionimpl.h"

namespace torchvision
{
class AlexNetImpl : public torch::nn::Module
{
	torch::nn::Sequential features, classifier;

public:
	AlexNetImpl(int num_classes = 1000);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(AlexNet);

}  // namespace torchvision

#endif  // ALEXNET_H
