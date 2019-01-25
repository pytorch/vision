#include <iostream>
#include "vision.h"

using namespace vision;

// TODO change xavier fill

template <typename T>
long params(T &M)
{
	long sum = 0;
	for (const auto &P : M->parameters())
	{
		long size = 1;
		for (const auto &S : P.sizes()) size *= S;
		sum += size;
	}

	return sum;
}

template <typename T>
void process(torch::Tensor X, std::string name)
{
	auto device = torch::kCPU;
	T M;
	M->to(device);

	M->train();
	M->forward(X.to(device));

	M->eval();
	M->forward(X.to(device));

	std::cout << name + " Done. Params: " << params(M) << " : "
			  << params(M) / 1000000 << std::endl;
}

#define PROCESS(M, X) process<M>(X, #M)

void init_weight(at::Tensor &weight, double stddev)
{
	auto temp = torch::fmod(torch::randn(weight.sizes()), 2);
	weight = temp.mul(stddev);
}

int main(int argc, const char *argv[])
{
	auto X = torch::rand({1, 3, 224, 224});

	PROCESS(AlexNet, X);
	PROCESS(VGG11, X);
	PROCESS(VGG13, X);
	PROCESS(VGG16, X);
	PROCESS(VGG19, X);
	PROCESS(VGG11BN, X);
	PROCESS(VGG13BN, X);
	PROCESS(VGG16BN, X);
	PROCESS(VGG19BN, X);
	PROCESS(ResNet18, X);
	PROCESS(ResNet34, X);
	PROCESS(ResNet50, X);
	PROCESS(ResNet101, X);
	PROCESS(ResNet152, X);
	PROCESS(SqueezeNet1_0, X);
	PROCESS(SqueezeNet1_1, X);
	PROCESS(DenseNet121, X);
	PROCESS(DenseNet169, X);
	PROCESS(DenseNet201, X);
	PROCESS(DenseNet161, X);

	X = torch::rand({2, 3, 299, 299});
	PROCESS(InceptionV3, X);

	return 0;
}
