#ifndef INCEPTION_H
#define INCEPTION_H

#include "visionimpl.h"

namespace torchvision
{
namespace inceptionimpl
{
class BasicConv2dImpl : public torch::nn::Module
{
	torch::nn::Conv2d conv;
	torch::nn::BatchNorm bn;

public:
	BasicConv2dImpl(int64_t in_channels, int64_t out_channels,
					torch::IntList kernel_size, torch::IntList padding = 0,
					torch::IntList stride = 1)
		: conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
				   .padding(padding)
				   .stride(stride)
				   .with_bias(false)),
		  bn(torch::nn::BatchNormOptions(out_channels).eps(0.001))
	{
		register_module("conv", conv);
		register_module("bn", bn);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = conv->forward(x);
		x = bn->forward(x);
		return torch::relu_(x);
	}
};

TORCH_MODULE(BasicConv2d);

class InceptionAImpl : public torch::nn::Module
{
	BasicConv2d branch1x1, branch5x5_1, branch5x5_2, branch3x3dbl_1,
		branch3x3dbl_2, branch3x3dbl_3, branch_pool;

public:
	InceptionAImpl(int64_t in_channels, int64_t pool_features)
		: branch1x1(in_channels, 64, 1),
		  branch5x5_1(in_channels, 48, 1),
		  branch5x5_2(48, 64, 5, 2),
		  branch3x3dbl_1(in_channels, 64, 1),
		  branch3x3dbl_2(64, 96, 3, 1),
		  branch3x3dbl_3(96, 96, 3, 1),
		  branch_pool(in_channels, pool_features, 1)
	{
		register_module("branch1x1", branch1x1);
		register_module("branch5x5_1", branch5x5_1);
		register_module("branch5x5_2", branch5x5_2);
		register_module("branch3x3dbl_1", branch3x3dbl_1);
		register_module("branch3x3dbl_2", branch3x3dbl_2);
		register_module("branch3x3dbl_3", branch3x3dbl_3);
		register_module("branch_pool", branch_pool);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		auto branch1x1 = this->branch1x1->forward(x);

		auto branch5x5 = this->branch5x5_1->forward(x);
		branch5x5 = this->branch5x5_2->forward(branch5x5);

		auto branch3x3dbl = this->branch3x3dbl_1->forward(x);
		branch3x3dbl = this->branch3x3dbl_2->forward(branch3x3dbl);
		branch3x3dbl = this->branch3x3dbl_3->forward(branch3x3dbl);

		auto branch_pool = torch::avg_pool2d(x, 3, 1, 1);
		branch_pool = this->branch_pool->forward(branch_pool);

		return torch::cat({branch1x1, branch5x5, branch3x3dbl, branch_pool}, 1);
	}
};

class InceptionBImpl : public torch::nn::Module
{
	BasicConv2d branch3x3, branch3x3dbl_1, branch3x3dbl_2, branch3x3dbl_3;

public:
	InceptionBImpl(int64_t in_channels)
		: branch3x3(in_channels, 384, 3, 0, 2),
		  branch3x3dbl_1(in_channels, 64, 1),
		  branch3x3dbl_2(64, 96, 3, 1),
		  branch3x3dbl_3(96, 96, 3, 0, 2)
	{
		register_module("branch3x3", branch3x3);
		register_module("branch3x3dbl_1", branch3x3dbl_1);
		register_module("branch3x3dbl_2", branch3x3dbl_2);
		register_module("branch3x3dbl_3", branch3x3dbl_3);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		auto branch3x3 = this->branch3x3->forward(x);

		auto branch3x3dbl = this->branch3x3dbl_1->forward(x);
		branch3x3dbl = this->branch3x3dbl_2->forward(branch3x3dbl);
		branch3x3dbl = this->branch3x3dbl_3->forward(branch3x3dbl);

		auto branch_pool = torch::max_pool2d(x, 3, 2);
		return torch::cat({branch3x3, branch3x3dbl, branch_pool}, 1);
	}
};

class InceptionCImpl : public torch::nn::Module
{
	BasicConv2d branch1x1{nullptr}, branch7x7_1{nullptr}, branch7x7_2{nullptr},
		branch7x7_3{nullptr}, branch7x7dbl_1{nullptr}, branch7x7dbl_2{nullptr},
		branch7x7dbl_3{nullptr}, branch7x7dbl_4{nullptr},
		branch7x7dbl_5{nullptr}, branch_pool{nullptr};

public:
	InceptionCImpl(int64_t in_channels, int64_t channels_7x7)
	{
		branch1x1 = BasicConv2d(in_channels, 192, 1);

		auto c7 = channels_7x7;
		branch7x7_1 = BasicConv2d(in_channels, c7, 1);
		branch7x7_2 =
			BasicConv2d(c7, c7, torch::IntList({1, 7}), torch::IntList({0, 3}));
		branch7x7_3 = BasicConv2d(c7, 192, torch::IntList({7, 1}),
								  torch::IntList({3, 0}));

		branch7x7dbl_1 = BasicConv2d(in_channels, c7, 1);
		branch7x7dbl_2 =
			BasicConv2d(c7, c7, torch::IntList({7, 1}), torch::IntList({3, 0}));
		branch7x7dbl_3 =
			BasicConv2d(c7, c7, torch::IntList({1, 7}), torch::IntList({0, 3}));
		branch7x7dbl_4 =
			BasicConv2d(c7, c7, torch::IntList({7, 1}), torch::IntList({3, 0}));
		branch7x7dbl_5 = BasicConv2d(c7, 192, torch::IntList({1, 7}),
									 torch::IntList({0, 3}));

		branch_pool = BasicConv2d(in_channels, 192, 1);

		register_module("branch1x1", branch1x1);
		register_module("branch7x7_1", branch7x7_1);
		register_module("branch7x7_2", branch7x7_2);
		register_module("branch7x7_3", branch7x7_3);
		register_module("branch7x7dbl_1", branch7x7dbl_1);
		register_module("branch7x7dbl_2", branch7x7dbl_2);
		register_module("branch7x7dbl_3", branch7x7dbl_3);
		register_module("branch7x7dbl_4", branch7x7dbl_4);
		register_module("branch7x7dbl_5", branch7x7dbl_5);
		register_module("branch_pool", branch_pool);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		auto branch1x1 = this->branch1x1->forward(x);

		auto branch7x7 = this->branch7x7_1->forward(x);
		branch7x7 = this->branch7x7_2->forward(branch7x7);
		branch7x7 = this->branch7x7_3->forward(branch7x7);

		auto branch7x7dbl = this->branch7x7dbl_1->forward(x);
		branch7x7dbl = this->branch7x7dbl_2->forward(branch7x7dbl);
		branch7x7dbl = this->branch7x7dbl_3->forward(branch7x7dbl);
		branch7x7dbl = this->branch7x7dbl_4->forward(branch7x7dbl);
		branch7x7dbl = this->branch7x7dbl_5->forward(branch7x7dbl);

		auto branch_pool = torch::avg_pool2d(x, 3, 1, 1);
		branch_pool = this->branch_pool->forward(branch_pool);

		return torch::cat({branch1x1, branch7x7, branch7x7dbl, branch_pool}, 1);
	}
};

class InceptionDImpl : public torch::nn::Module
{
	BasicConv2d branch3x3_1, branch3x3_2, branch7x7x3_1, branch7x7x3_2,
		branch7x7x3_3, branch7x7x3_4;

public:
	InceptionDImpl(int64_t in_channels)
		: branch3x3_1(in_channels, 192, 1),
		  branch3x3_2(192, 320, 3, 0, 2),
		  branch7x7x3_1(in_channels, 192, 1),
		  branch7x7x3_2(192, 192, torch::IntList({1, 7}),
						torch::IntList({0, 3})),
		  branch7x7x3_3(192, 192, torch::IntList({7, 1}),
						torch::IntList({3, 0})),
		  branch7x7x3_4(192, 192, 3, 0, 2)

	{
		register_module("branch3x3_1", branch3x3_1);
		register_module("branch3x3_2", branch3x3_2);
		register_module("branch7x7x3_1", branch7x7x3_1);
		register_module("branch7x7x3_2", branch7x7x3_2);
		register_module("branch7x7x3_3", branch7x7x3_3);
		register_module("branch7x7x3_4", branch7x7x3_4);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		auto branch3x3 = this->branch3x3_1->forward(x);
		branch3x3 = this->branch3x3_2->forward(branch3x3);

		auto branch7x7x3 = this->branch7x7x3_1->forward(x);
		branch7x7x3 = this->branch7x7x3_2->forward(branch3x3);
		branch7x7x3 = this->branch7x7x3_3->forward(branch3x3);
		branch7x7x3 = this->branch7x7x3_4->forward(branch3x3);

		auto branch_pool = torch::max_pool2d(x, 3, 2);
		return torch::cat({branch3x3, branch7x7x3, branch_pool}, 1);
	}
};

class InceptionEImpl : public torch::nn::Module
{
	BasicConv2d branch1x1, branch3x3_1, branch3x3_2a, branch3x3_2b,
		branch3x3dbl_1, branch3x3dbl_2, branch3x3dbl_3a, branch3x3dbl_3b,
		branch_pool;

public:
	InceptionEImpl(int64_t in_channels)
		: branch1x1(in_channels, 320, 1),
		  branch3x3_1(in_channels, 384, 1),
		  branch3x3_2a(384, 384, torch::IntList({1, 3}),
					   torch::IntList({0, 1})),
		  branch3x3_2b(384, 384, torch::IntList({3, 1}),
					   torch::IntList({1, 0})),
		  branch3x3dbl_1(in_channels, 448, 1),
		  branch3x3dbl_2(448, 384, 3, 1),
		  branch3x3dbl_3a(384, 384, torch::IntList({1, 3}),
						  torch::IntList({0, 1})),
		  branch3x3dbl_3b(384, 384, torch::IntList({3, 1}),
						  torch::IntList({1, 0})),
		  branch_pool(in_channels, 192, 1)
	{
		register_module("branch1x1", branch1x1);
		register_module("branch3x3_1", branch3x3_1);
		register_module("branch3x3_2a", branch3x3_2a);
		register_module("branch3x3_2b", branch3x3_2b);
		register_module("branch3x3dbl_1", branch3x3dbl_1);
		register_module("branch3x3dbl_2", branch3x3dbl_2);
		register_module("branch3x3dbl_3a", branch3x3dbl_3a);
		register_module("branch3x3dbl_3b", branch3x3dbl_3b);
		register_module("branch_pool", branch_pool);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		auto branch1x1 = this->branch1x1->forward(x);

		auto branch3x3 = this->branch3x3_1->forward(x);
		branch3x3 = torch::cat(
			{
				this->branch3x3_2a->forward(branch3x3),
				this->branch3x3_2b->forward(branch3x3),
			},
			1);

		auto branch3x3dbl = this->branch3x3dbl_1->forward(x);
		branch3x3dbl = this->branch3x3dbl_2->forward(branch3x3dbl);
		branch3x3dbl =
			torch::cat({this->branch3x3dbl_3a->forward(branch3x3dbl),
						this->branch3x3dbl_3b->forward(branch3x3dbl)},
					   1);

		auto branch_pool = torch::avg_pool2d(x, 3, 1, 1);
		return torch::cat({branch1x1, branch3x3, branch3x3dbl, branch_pool}, 1);
	}
};

class InceptionAuxImpl : public torch::nn::Module
{
	BasicConv2d conv0, conv1;
	torch::nn::Linear fc;

public:
	InceptionAuxImpl(int64_t in_channels, int64_t num_classes)
		: conv0(BasicConv2d(in_channels, 128, 1)),
		  conv1(BasicConv2d(128, 768, 5)),
		  fc(768, num_classes)
	{
		// TODO set these
		// self.conv1.stddev = 0.01
		// self.fc.stddev = 0.001

		register_module("conv0", conv0);
		register_module("conv1", conv1);
		register_module("fc", fc);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::avg_pool2d(x, 5, 3);
		x = conv0->forward(x);
		x = conv1->forward(x);
		x = x.view({x.size(0), -1});
		x = fc->forward(x);
		return x;
	}
};

TORCH_MODULE(InceptionA);
TORCH_MODULE(InceptionB);
TORCH_MODULE(InceptionC);
TORCH_MODULE(InceptionD);
TORCH_MODULE(InceptionE);
TORCH_MODULE(InceptionAux);

}  // namespace inceptionimpl

class Inception_V3Impl : public torch::nn::Module
{
	bool aux_logits, transform_input;
	inceptionimpl::BasicConv2d Conv2d_1a_3x3{nullptr}, Conv2d_2a_3x3{nullptr},
		Conv2d_2b_3x3{nullptr}, Conv2d_3b_1x1{nullptr}, Conv2d_4a_3x3{nullptr};
	inceptionimpl::InceptionA Mixed_5b{nullptr}, Mixed_5c{nullptr},
		Mixed_5d{nullptr};
	inceptionimpl::InceptionB Mixed_6a{nullptr};
	inceptionimpl::InceptionC Mixed_6b{nullptr}, Mixed_6c{nullptr},
		Mixed_6d{nullptr}, Mixed_6e{nullptr};
	inceptionimpl::InceptionD Mixed_7a{nullptr};
	inceptionimpl::InceptionE Mixed_7b{nullptr}, Mixed_7c{nullptr};
	torch::nn::Linear fc{nullptr};

	inceptionimpl::InceptionAux AuxLogits{nullptr};

public:
	Inception_V3Impl(int64_t num_classes = 1000, bool aux_logits = true,
					 bool transform_input = false)
		: aux_logits(aux_logits), transform_input(transform_input)
	{
		Conv2d_1a_3x3 = inceptionimpl::BasicConv2d(3, 32, 3, 0, 2);
		Conv2d_2a_3x3 = inceptionimpl::BasicConv2d(32, 32, 3);
		Conv2d_2b_3x3 = inceptionimpl::BasicConv2d(32, 64, 3, 1);
		Conv2d_3b_1x1 = inceptionimpl::BasicConv2d(64, 80, 1);
		Conv2d_4a_3x3 = inceptionimpl::BasicConv2d(80, 192, 3);

		Mixed_5b = inceptionimpl::InceptionA(192, 32);
		Mixed_5c = inceptionimpl::InceptionA(256, 64);
		Mixed_5d = inceptionimpl::InceptionA(288, 64);

		Mixed_6a = inceptionimpl::InceptionB(288);
		Mixed_6b = inceptionimpl::InceptionC(768, 128);
		Mixed_6c = inceptionimpl::InceptionC(768, 160);
		Mixed_6d = inceptionimpl::InceptionC(768, 160);
		Mixed_6e = inceptionimpl::InceptionC(768, 192);

		if (aux_logits)
			AuxLogits = inceptionimpl::InceptionAux(768, num_classes);

		Mixed_7a = inceptionimpl::InceptionD(768);
		Mixed_7b = inceptionimpl::InceptionE(1280);
		Mixed_7c = inceptionimpl::InceptionE(2048);

		fc = torch::nn::Linear(2048, num_classes);

		register_module("Conv2d_1a_3x3", Conv2d_1a_3x3);
		register_module("Conv2d_2a_3x3", Conv2d_2a_3x3);
		register_module("Conv2d_2b_3x3", Conv2d_2b_3x3);
		register_module("Conv2d_3b_1x1", Conv2d_3b_1x1);
		register_module("Conv2d_4a_3x3", Conv2d_4a_3x3);
		register_module("Mixed_5b", Mixed_5b);
		register_module("Mixed_5c", Mixed_5c);
		register_module("Mixed_5d", Mixed_5d);
		register_module("Mixed_6a", Mixed_6a);
		register_module("Mixed_6b", Mixed_6b);
		register_module("Mixed_6c", Mixed_6c);
		register_module("Mixed_6d", Mixed_6d);
		register_module("Mixed_6e", Mixed_6e);

		if (!AuxLogits.is_empty()) register_module("AuxLogits", AuxLogits);

		register_module("Mixed_7a", Mixed_7a);
		register_module("Mixed_7b", Mixed_7b);
		register_module("Mixed_7c", Mixed_7c);
		register_module("fc", fc);

		// TODO
		//		for m in self.modules():
		//            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		//                import scipy.stats as stats
		//                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
		//                X = stats.truncnorm(-2, 2, scale=stddev)
		//                values = torch.Tensor(X.rvs(m.weight.numel()))
		//                values = values.view(m.weight.size())
		//                m.weight.data.copy_(values)
		//            elif isinstance(m, nn.BatchNorm2d):
		//                nn.init.constant_(m.weight, 1)
		//                nn.init.constant_(m.bias, 0)
	}

	torch::Tensor forward(torch::Tensor x)
	{
		if (transform_input)
		{
		}
	}
};

TORCH_MODULE(Inception_V3);

}  // namespace torchvision

#endif  // INCEPTION_H
