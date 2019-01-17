#include "vgg.h"

void torchvision::VGGImpl::initializeWeights()
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

torchvision::VGGImpl::VGGImpl(torch::nn::Sequential features, int classes,
							  bool initWeights)
{
    // clang-format off
    classifier = torch::nn::Sequential(
                torch::nn::Linear(512 * 7 * 7, 4096),
				visionimpl::Relu(true),
                torch::nn::Dropout(),
                torch::nn::Linear(4096, 4096),
				visionimpl::Relu(true),
                torch::nn::Dropout(),
                torch::nn::Linear(4096, classes));
    // clang-format on

    this->features = features;

    register_module("features", features);
    register_module("classifier", classifier);

    if (initWeights) initializeWeights();
}

torch::Tensor torchvision::VGGImpl::forward(at::Tensor X)
{
    X = features->forward(X);
    X = X.view({X.size(0), -1});
    X = classifier->forward(X);
    return X;
}

torch::nn::Sequential torchvision::makeLayers(const std::vector<int> &cfg,
											  bool batchNorm)
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
            if (batchNorm) seq->push_back(torch::nn::BatchNorm(V));
			seq->push_back(visionimpl::Relu(true));

            channels = V;
        }
    }

    return seq;
}

torchvision::VGG11Impl::VGG11Impl(int classes, bool initWeights)
	: VGGImpl(makeLayers(
				  {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}),
			  classes, initWeights)
{
}

torchvision::VGG13Impl::VGG13Impl(int classes, bool initWeights)
	: VGGImpl(makeLayers({64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1,
						  512, 512, -1}),
			  classes, initWeights)
{
}

torchvision::VGG16Impl::VGG16Impl(int classes, bool initWeights)
	: VGGImpl(makeLayers({64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512,
						  512, -1, 512, 512, 512, -1}),
			  classes, initWeights)
{
}

torchvision::VGG19Impl::VGG19Impl(int classes, bool initWeights)
	: VGGImpl(makeLayers({64,  64,  -1,  128, 128, -1,  256, 256, 256, 256, -1,
						  512, 512, 512, 512, -1,  512, 512, 512, 512, -1}),
			  classes, initWeights)
{
}

torchvision::VGG11BNImpl::VGG11BNImpl(int classes, bool initWeights)
	: VGGImpl(makeLayers(
				  {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1},
				  true),
			  classes, initWeights)
{
}

torchvision::VGG13BNImpl::VGG13BNImpl(int classes, bool initWeights)
	: VGGImpl(makeLayers({64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1,
						  512, 512, -1},
						 true),
			  classes, initWeights)
{
}

torchvision::VGG16BNImpl::VGG16BNImpl(int classes, bool initWeights)
	: VGGImpl(makeLayers({64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512,
						  512, -1, 512, 512, 512, -1},
						 true),
			  classes, initWeights)
{
}

torchvision::VGG19BNImpl::VGG19BNImpl(int classes, bool initWeights)
	: VGGImpl(makeLayers({64,  64,  -1,  128, 128, -1,  256, 256, 256, 256, -1,
						  512, 512, 512, 512, -1,  512, 512, 512, 512, -1},
						 true),
			  classes, initWeights)
{
}
