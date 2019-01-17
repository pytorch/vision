#include <iostream>
#include "vision.h"

using namespace torchvision;

// TODO change num_classes from int to int64_t

int main()
{
	auto X = torch::rand({1, 3, 224, 224});

	AlexNet A;
	A->forward(X);
	std::cout << "AlexNet Done" << std::endl;

	VGG11 V11;
	V11->forward(X);
	std::cout << "VGG11 Done" << std::endl;

	VGG13 V13;
	V13->forward(X);
	std::cout << "VGG13 Done" << std::endl;

	VGG16 V16;
	V16->forward(X);
	std::cout << "VGG16 Done" << std::endl;

	VGG19 V19;
	V19->forward(X);
	std::cout << "VGG19 Done" << std::endl;

	VGG11BN V11BN;
	V11BN->forward(X);
	std::cout << "VGG11BN Done" << std::endl;

	VGG13BN V13BN;
	V13BN->forward(X);
	std::cout << "VGG13BN Done" << std::endl;

	VGG16BN V16BN;
	V16BN->forward(X);
	std::cout << "VGG16BN Done" << std::endl;

	VGG19BN V19BN;
	V19BN->forward(X);
	std::cout << "VGG19BN Done" << std::endl;

	ResNet18 R18;
	R18->forward(X);
	std::cout << "ResNet18 Done" << std::endl;

	ResNet34 R34;
	R34->forward(X);
	std::cout << "ResNet34 Done" << std::endl;

	ResNet50 R50;
	R50->forward(X);
	std::cout << "ResNet50 Done" << std::endl;

	ResNet101 R101;
	R101->forward(X);
	std::cout << "ResNet101 Done" << std::endl;

	ResNet152 R152;
	R152->forward(X);
	std::cout << "ResNet152 Done" << std::endl;

	SqueezeNet1_0 S10;
	S10->forward(X);
	std::cout << "SqueezeNet1.0 Done" << std::endl;

	SqueezeNet1_1 S11;
	S11->forward(X);
	std::cout << "SqueezeNet1.1 Done" << std::endl;

	return 0;
}
