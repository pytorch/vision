#include "vgg.h"

#include <map>
#include "visionimpl.h"

namespace torchvision
{
torch::nn::Sequential makeLayers(const std::vector<int> &cfg,
								 bool batch_norm = false)
{
	torch::nn::Sequential seq;
	auto channels = 3;

	for (const auto &V : cfg)
	{
		if (V <= -1)
			seq->push_back(visionimpl::MaxPool2D(2, 2));
		else
		{
			seq->push_back(torch::nn::Conv2d(
				torch::nn::Conv2dOptions(channels, V, 3).padding(1)));

			if (batch_norm) seq->push_back(torch::nn::BatchNorm(V));
			seq->push_back(visionimpl::Relu(true));

			channels = V;
		}
	}

	return seq;
}

void VGGImpl::_initialize_weights()
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

VGGImpl::VGGImpl(torch::nn::Sequential features, int64_t num_classes,
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

torch::Tensor VGGImpl::forward(torch::Tensor x)
{
	x = features->forward(x);
	x = x.view({x.size(0), -1});
	x = classifier->forward(x);
	return x;
}

// clang-format off
static std::map<char, std::vector<int>> cfg = {
	{'A', {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
	{'B', {64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
	{'D', {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1}},
	{'E', {64,  64,  -1,  128, 128, -1,  256, 256, 256, 256, -1, 512, 512, 512, 512, -1,  512, 512, 512, 512, -1}}};
// clang-format on

VGG11Impl::VGG11Impl(int64_t num_classes, bool initialize_weights)
	: VGGImpl(makeLayers(cfg['A']), num_classes, initialize_weights)
{
}

VGG13Impl::VGG13Impl(int64_t num_classes, bool initWeights)
	: VGGImpl(makeLayers(cfg['B']), num_classes, initWeights)
{
}

VGG16Impl::VGG16Impl(int64_t num_classes, bool initWeights)
	: VGGImpl(makeLayers(cfg['D']), num_classes, initWeights)
{
}

VGG19Impl::VGG19Impl(int64_t num_classes, bool initialize_weights)
	: VGGImpl(makeLayers(cfg['E']), num_classes, initialize_weights)
{
}

VGG11BNImpl::VGG11BNImpl(int64_t num_classes, bool initWeights)
	: VGGImpl(makeLayers(cfg['A'], true), num_classes, initWeights)
{
}

VGG13BNImpl::VGG13BNImpl(int64_t num_classes, bool initialize_weights)
	: VGGImpl(makeLayers(cfg['B'], true), num_classes, initialize_weights)
{
}

VGG16BNImpl::VGG16BNImpl(int64_t num_classes, bool initWeights)
	: VGGImpl(makeLayers(cfg['D'], true), num_classes, initWeights)
{
}

VGG19BNImpl::VGG19BNImpl(int64_t num_classes, bool initWeights)
	: VGGImpl(makeLayers(cfg['E'], true), num_classes, initWeights)
{
}

}  // namespace torchvision
