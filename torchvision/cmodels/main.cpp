#include <iostream>
#include "vision.h"

using namespace torchvision;

// TODO change num_classes from int to int64_t
// TODO in some classes I have initialized weights of submodules before
// registering them. this is wrong. fix it

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
	T M;
	M->forward(X);
	std::cout << name + " Done. Params: " << params(M) << " : "
			  << params(M) / 1000000 << std::endl;
}

#define PROCESS(M, X) process<M>(X, #M)

int main()
{
	auto X = torch::rand({1, 3, 224, 224});

	//	AlexNet A;
	//	A->forward(X);
	//	std::cout << "AlexNet Done. #Params: " << params(A) << std::endl;

	//	VGG11 V11;
	//	V11->forward(X);
	//	std::cout << "VGG11 Done. #Params: " << params(V11) << std::endl;

	//	VGG13 V13;
	//	V13->forward(X);
	//	std::cout << "VGG13 Done. #Params: " << params(V13) << std::endl;

	//	VGG16 V16;
	//	V16->forward(X);
	//	std::cout << "VGG16 Done. #Params: " << params(V16) << std::endl;

	//	VGG19 V19;
	//	V19->forward(X);
	//	std::cout << "VGG19 Done. #Params: " << params(V19) << std::endl;

	//	VGG11BN V11BN;
	//	V11BN->forward(X);
	//	std::cout << "VGG11BN Done. #Params: " << params(V11BN) << std::endl;

	//	VGG13BN V13BN;
	//	V13BN->forward(X);
	//	std::cout << "VGG13BN Done. #Params: " << params(V13BN) << std::endl;

	//	VGG16BN V16BN;
	//	V16BN->forward(X);
	//	std::cout << "VGG16BN Done. #Params: " << params(V16BN) << std::endl;

	//	VGG19BN V19BN;
	//	V19BN->forward(X);
	//	std::cout << "VGG19BN Done. #Params: " << params(V19BN) << std::endl;

	//	ResNet18 R18;
	//	R18->forward(X);
	//	std::cout << "ResNet18 Done. #Params: " << params(R18) << std::endl;

	//	ResNet34 R34;
	//	R34->forward(X);
	//	std::cout << "ResNet34 Done. #Params: " << params(R34) << std::endl;

	//	ResNet50 R50;
	//	R50->forward(X);
	//	std::cout << "ResNet50 Done. #Params: " << params(R50) << std::endl;

	//	ResNet101 R101;
	//	R101->forward(X);
	//	std::cout << "ResNet101 Done. #Params: " << params(R101) << std::endl;

	//	ResNet152 R152;
	//	R152->forward(X);
	//	std::cout << "ResNet152 Done. #Params: " << params(R152) << std::endl;

	//	SqueezeNet1_0 S10;
	//	S10->forward(X);
	//	std::cout << "SqueezeNet1.0 Done. #Params: " << params(S10) <<
	// std::endl;

	//	SqueezeNet1_1 S11;
	//	S11->forward(X);
	//	std::cout << "SqueezeNet1.1 Done. #Params: " << params(S11) <<
	// std::endl;

	//	DenseNet121 D121;
	//	D121->forward(X);
	//	std::cout << "DenseNet121 Done. #Params: " << params(D121) << std::endl;

	//	DenseNet169 D169;
	//	D169->forward(X);
	//	std::cout << "DenseNet169 Done. #Params: " << params(D169) << std::endl;

	//	DenseNet201 D201;
	//	D201->forward(X);
	//	std::cout << "DenseNet201 Done. #Params: " << params(D201) << std::endl;

	//	DenseNet161 D161;
	//	D161->forward(X);
	//	std::cout << "DenseNet161 Done. #Params: " << params(D161) << std::endl;

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

	return 0;
}
