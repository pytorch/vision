#include "vgg.h"

void torchvision::VGGImpl::_initialize_weights()
{
    for (auto &module : modules(false))
    {
        if (torch::nn::Conv2dImpl *M =
                dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
        {
            torch::nn::init::xavier_normal_(M->weight);
            torch::nn::init::constant_(M->bias, 0);
        }
        else if (torch::nn::BatchNormImpl *M =
					 dynamic_cast<torch::nn::BatchNormImpl *>(module.get()))
        {
            torch::nn::init::constant_(M->weight, 1);
            torch::nn::init::constant_(M->bias, 0);
        }
        else if (torch::nn::LinearImpl *M =
					 dynamic_cast<torch::nn::LinearImpl *>(module.get()))
        {
            torch::nn::init::normal_(M->weight, 0, 0.01);
            torch::nn::init::normal_(M->bias, 0);
        }
    }
}

torchvision::VGGImpl::VGGImpl(torch::nn::Sequential features, int num_classes,
							  bool initialize_weights)
{
    // clang-format off
    classifier = torch::nn::Sequential(
                torch::nn::Linear(512 * 7 * 7, 4096),
				visionimpl::Relu(true),
                torch::nn::Dropout(),
                torch::nn::Linear(4096, 4096),
				visionimpl::Relu(true),
                torch::nn::Dropout(),
				torch::nn::Linear(4096, num_classes));
    // clang-format on

    this->features = features;

	register_module("features", this->features);
    register_module("classifier", classifier);

	if (initialize_weights) _initialize_weights();
}

torch::Tensor torchvision::VGGImpl::forward(at::Tensor x)
{
	x = features->forward(x);
	x = x.view({x.size(0), -1});
	x = classifier->forward(x);
	return x;
}

torch::nn::Sequential torchvision::makeLayers(const std::vector<int> &cfg,
											  bool batch_norm)
{
    torch::nn::Sequential seq;
    auto channels = 3;

    for (const auto &V : cfg)
    {
        if (V <= -1)
			seq->push_back(visionimpl::MaxPool(2, 2));
        else
        {
			seq->push_back(visionimpl::Conv(channels, V, 3, 1));
			if (batch_norm) seq->push_back(torch::nn::BatchNorm(V));
			seq->push_back(visionimpl::Relu(true));

            channels = V;
        }
    }

    return seq;
}

torchvision::VGG11Impl::VGG11Impl(int num_classes, bool initialize_weights)
	: VGGImpl(makeLayers(
				  {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}),
			  num_classes, initialize_weights)
{
}

torchvision::VGG13Impl::VGG13Impl(int num_classes, bool initWeights)
	: VGGImpl(makeLayers({64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1,
						  512, 512, -1}),
			  num_classes, initWeights)
{
}

torchvision::VGG16Impl::VGG16Impl(int num_classes, bool initWeights)
	: VGGImpl(makeLayers({64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512,
						  512, -1, 512, 512, 512, -1}),
			  num_classes, initWeights)
{
}

torchvision::VGG19Impl::VGG19Impl(int num_classes, bool initialize_weights)
	: VGGImpl(makeLayers({64,  64,  -1,  128, 128, -1,  256, 256, 256, 256, -1,
						  512, 512, 512, 512, -1,  512, 512, 512, 512, -1}),
			  num_classes, initialize_weights)
{
}

torchvision::VGG11BNImpl::VGG11BNImpl(int num_classes, bool initWeights)
	: VGGImpl(makeLayers(
				  {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1},
				  true),
			  num_classes, initWeights)
{
}

torchvision::VGG13BNImpl::VGG13BNImpl(int num_classes, bool initialize_weights)
	: VGGImpl(makeLayers({64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1,
						  512, 512, -1},
						 true),
			  num_classes, initialize_weights)
{
}

torchvision::VGG16BNImpl::VGG16BNImpl(int num_classes, bool initWeights)
	: VGGImpl(makeLayers({64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512,
						  512, -1, 512, 512, 512, -1},
						 true),
			  num_classes, initWeights)
{
}

torchvision::VGG19BNImpl::VGG19BNImpl(int num_classes, bool initWeights)
	: VGGImpl(makeLayers({64,  64,  -1,  128, 128, -1,  256, 256, 256, 256, -1,
						  512, 512, 512, 512, -1,  512, 512, 512, 512, -1},
						 true),
			  num_classes, initWeights)
{
}
