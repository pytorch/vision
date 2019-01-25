#ifndef SQUEEZENET_H
#define SQUEEZENET_H

#include <torch/torch.h>

namespace vision
{
class SqueezeNetImpl : public torch::nn::Module
{
	int64_t num_classes;
	torch::nn::Sequential features{nullptr}, classifier{nullptr};

public:
	SqueezeNetImpl(double version = 1.0, int64_t num_classes = 1000);

	torch::Tensor forward(torch::Tensor x);
};

// SqueezeNet model architecture from the "SqueezeNet: AlexNet-level
// accuracy with 50x fewer parameters and <0.5MB model size"
// <https://arxiv.org/abs/1602.07360> paper.
class SqueezeNet1_0Impl : public SqueezeNetImpl
{
public:
	SqueezeNet1_0Impl(int64_t num_classes = 1000);
};

// SqueezeNet 1.1 model from the official SqueezeNet repo
// <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>.
// SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
// than SqueezeNet 1.0, without sacrificing accuracy.
class SqueezeNet1_1Impl : public SqueezeNetImpl
{
public:
	SqueezeNet1_1Impl(int64_t num_classes = 1000);
};

TORCH_MODULE(SqueezeNet);
TORCH_MODULE(SqueezeNet1_0);
TORCH_MODULE(SqueezeNet1_1);

}  // namespace torchvision

#endif  // SQUEEZENET_H
