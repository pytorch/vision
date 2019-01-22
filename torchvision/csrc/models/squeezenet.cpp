#include "squeezenet.h"

#include <limits>
#include "visionimpl.h"

namespace torchvision
{
class Fire : public torch::nn::Module
{
	torch::nn::Conv2d squeeze, expand1x1, expand3x3;

public:
	Fire(int64_t inplanes, int64_t squeeze_planes, int64_t expand1x1_planes,
		 int64_t expand3x3_planes)
		: squeeze(torch::nn::Conv2dOptions(inplanes, squeeze_planes, 1)),
		  expand1x1(
			  torch::nn::Conv2dOptions(squeeze_planes, expand1x1_planes, 1)),
		  expand3x3(
			  torch::nn::Conv2dOptions(squeeze_planes, expand3x3_planes, 3)
				  .padding(1))
	{
		register_module("squeeze", squeeze);
		register_module("expand1x1", expand1x1);
		register_module("expand3x3", expand3x3);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(squeeze->forward(x));
		return torch::cat({torch::relu(expand1x1->forward(x)),
						   torch::relu(expand3x3->forward(x))},
						  1);
	}
};

SqueezeNetImpl::SqueezeNetImpl(double version, int64_t num_classes)
	: num_classes(num_classes)
{
	auto double_compare = [](double a, double b) {
		return double(std::abs(a - b)) < std::numeric_limits<double>::epsilon();
	};

	if (double_compare(version, 1.0))
	{
		// clang-format off
		features = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 96, 7).stride(2)),
			visionimpl::Relu(true),
			visionimpl::MaxPool2D(3, 2, true),
			Fire(96, 16, 64, 64),
			Fire(128, 16, 64, 64),
			Fire(128, 32, 128, 128),
			visionimpl::MaxPool2D(3, 2, true),
			Fire(256, 32, 128, 128),
			Fire(256, 48, 192, 192),
			Fire(384, 48, 192, 192),
			Fire(384, 64, 256, 256),
			visionimpl::MaxPool2D(3, 2, true),
			Fire(512, 64, 256, 256));
		// clang-format on
	}
	else if (double_compare(version, 1.1))
	{
		// clang-format off
		features = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(2)),
			visionimpl::Relu(true),
			visionimpl::MaxPool2D(3, 2, true),
			Fire(64, 16, 64, 64),
			Fire(128, 16, 64, 64),
			visionimpl::MaxPool2D(3, 2, true),
			Fire(128, 32, 128, 128),
			Fire(256, 32, 128, 128),
			visionimpl::MaxPool2D(3, 2, true),
			Fire(256, 48, 192, 192),
			Fire(384, 48, 192, 192),
			Fire(384, 64, 256, 256),
			Fire(512, 64, 256, 256));
		// clang-format on
	}
	else
	{
		std::cerr << "Wrong version number is passed th SqueeseNet constructor!"
				  << std::endl;
		assert(false);
	}

	auto final_conv =
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, num_classes, 1));

	// clang-format off
	classifier = torch::nn::Sequential(
		torch::nn::Dropout(0.5),
		final_conv,
		visionimpl::Relu(true),
		visionimpl::AdaptiveAvgPool2D(torch::IntList({1, 1})));
	// clang-format on

	register_module("features", features);
	register_module("classifier", classifier);

	for (auto &module : modules(false))
		if (torch::nn::Conv2dImpl *M =
				dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
		{
			if (M == final_conv.get())
				torch::nn::init::normal_(M->weight, 0.0, 0.01);
			else
				torch::nn::init::xavier_uniform_(M->weight);

			if (M->options.with_bias()) torch::nn::init::constant_(M->bias, 0);
		}
}

torch::Tensor SqueezeNetImpl::forward(torch::Tensor x)
{
	x = features->forward(x);
	x = classifier->forward(x);
	return x.view({x.size(0), num_classes});
}

SqueezeNet1_0Impl::SqueezeNet1_0Impl(int64_t num_classes)
	: SqueezeNetImpl(1.0, num_classes)
{
}

SqueezeNet1_1Impl::SqueezeNet1_1Impl(int64_t num_classes)
	: SqueezeNetImpl(1.1, num_classes)
{
}

}  // namespace torchvision
