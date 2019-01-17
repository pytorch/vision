#include "squeezenet.h"

torchvision::squeezenetimpl::Fire::Fire(int64_t inplanes,
										int64_t squeeze_planes,
										int64_t expand1x1_planes,
										int64_t expand3x3_planes)
	: inplanes(inplanes),
	  squeeze(torch::nn::Conv2dOptions(inplanes, squeeze_planes, 1)),
	  expand1x1(torch::nn::Conv2dOptions(squeeze_planes, expand1x1_planes, 1)),
	  expand3x3(torch::nn::Conv2dOptions(squeeze_planes, expand3x3_planes, 3)
					.padding(1))
{
	register_module("squeeze", squeeze);
	register_module("expand1x1", expand1x1);
	register_module("expand3x3", expand3x3);
}

torch::Tensor torchvision::squeezenetimpl::Fire::forward(at::Tensor x)
{
	x = torch::relu(squeeze->forward(x));
	return torch::cat({torch::relu(expand1x1->forward(x)),
					   torch::relu(expand3x3->forward(x))},
					  1);
}

torchvision::SqueezeNetImpl::SqueezeNetImpl(double version, int num_classes)
	: num_classes(num_classes)
{
	// TODO change double compare

	if (version == 1.0)
	{
		features = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 96, 7).stride(2)),
			visionimpl::Relu(true), visionimpl::MaxPool(3, 2, true),
			squeezenetimpl::Fire(96, 16, 64, 64),
			squeezenetimpl::Fire(128, 16, 64, 64),
			squeezenetimpl::Fire(128, 32, 128, 128),
			visionimpl::MaxPool(3, 2, true),
			squeezenetimpl::Fire(256, 32, 128, 128),
			squeezenetimpl::Fire(256, 48, 192, 192),
			squeezenetimpl::Fire(384, 48, 192, 192),
			squeezenetimpl::Fire(384, 64, 256, 256),
			visionimpl::MaxPool(3, 2, true),
			squeezenetimpl::Fire(512, 64, 256, 256));
	}
	else if (version == 1.1)
	{
		features = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(2)),
			visionimpl::Relu(true), visionimpl::MaxPool(3, 2, true),
			squeezenetimpl::Fire(64, 16, 64, 64),
			squeezenetimpl::Fire(128, 16, 64, 64),
			visionimpl::MaxPool(3, 2, true),
			squeezenetimpl::Fire(128, 32, 128, 128),
			squeezenetimpl::Fire(256, 32, 128, 128),
			visionimpl::MaxPool(3, 2, true),
			squeezenetimpl::Fire(256, 48, 192, 192),
			squeezenetimpl::Fire(384, 48, 192, 192),
			squeezenetimpl::Fire(384, 64, 256, 256),
			squeezenetimpl::Fire(512, 64, 256, 256));
	}

	auto final_conv =
		torch::nn::Conv2d(torch::nn::Conv2dOptions(512, num_classes, 1));

	classifier = torch::nn::Sequential(
		torch::nn::Dropout(0.5), final_conv, visionimpl::Relu(true),
		visionimpl::AdaptiveAvgPool2D(torch::IntList({1, 1})));

	for (auto &module : modules(false))
	{
		if (torch::nn::Conv2dImpl *M =
				dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
		{
			if (M == final_conv.get())
				torch::nn::init::normal_(M->weight, 0.0, 0.01);
			else
				torch::nn::init::xavier_uniform_(
					M->weight);  // TODO this should be kaiming

			if (M->options.with_bias()) torch::nn::init::constant_(M->bias, 0);
		}
	}

	register_module("features", features);
	register_module("classifier", classifier);
}

torch::Tensor torchvision::SqueezeNetImpl::forward(at::Tensor x)
{
	x = features->forward(x);
	x = classifier->forward(x);
	return x.view({x.size(0), num_classes});
}

torchvision::SqueezeNet1_0Impl::SqueezeNet1_0Impl(int num_classes)
	: SqueezeNetImpl(1.0, num_classes)
{
}

torchvision::SqueezeNet1_1Impl::SqueezeNet1_1Impl(int num_classes)
	: SqueezeNetImpl(1.1, num_classes)
{
}
