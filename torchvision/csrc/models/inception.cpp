#include "inception.h"

namespace vision {
namespace models {

using Options = torch::nn::Conv2dOptions;

namespace _inceptionimpl {
BasicConv2dImpl::BasicConv2dImpl(
    torch::nn::Conv2dOptions options,
    double std_dev) {
  options.bias(false);
  conv = torch::nn::Conv2d(options);
  bn = torch::nn::BatchNorm(
      torch::nn::BatchNormOptions(options.out_channels()).eps(0.001));

  register_module("conv", conv);
  register_module("bn", bn);

  torch::nn::init::normal_(
      conv->weight,
      0,
      std_dev); // Note: used instead of truncated normal initialization

  torch::nn::init::constant_(bn->weight, 1);
  torch::nn::init::constant_(bn->bias, 0);
}

torch::Tensor BasicConv2dImpl::forward(torch::Tensor x) {
  x = conv->forward(x);
  x = bn->forward(x);
  return torch::relu_(x);
}

InceptionAImpl::InceptionAImpl(int64_t in_channels, int64_t pool_features)
    : branch1x1(Options(in_channels, 64, 1)),
      branch5x5_1(Options(in_channels, 48, 1)),
      branch5x5_2(Options(48, 64, 5).padding(2)),
      branch3x3dbl_1(Options(in_channels, 64, 1)),
      branch3x3dbl_2(Options(64, 96, 3).padding(1)),
      branch3x3dbl_3(Options(96, 96, 3).padding(1)),
      branch_pool(Options(in_channels, pool_features, 1)) {
  register_module("branch1x1", branch1x1);
  register_module("branch5x5_1", branch5x5_1);
  register_module("branch5x5_2", branch5x5_2);
  register_module("branch3x3dbl_1", branch3x3dbl_1);
  register_module("branch3x3dbl_2", branch3x3dbl_2);
  register_module("branch3x3dbl_3", branch3x3dbl_3);
  register_module("branch_pool", branch_pool);
}

torch::Tensor InceptionAImpl::forward(torch::Tensor x) {
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

InceptionBImpl::InceptionBImpl(int64_t in_channels)
    : branch3x3(Options(in_channels, 384, 3).stride(2)),
      branch3x3dbl_1(Options(in_channels, 64, 1)),
      branch3x3dbl_2(Options(64, 96, 3).padding(1)),
      branch3x3dbl_3(Options(96, 96, 3).stride(2)) {
  register_module("branch3x3", branch3x3);
  register_module("branch3x3dbl_1", branch3x3dbl_1);
  register_module("branch3x3dbl_2", branch3x3dbl_2);
  register_module("branch3x3dbl_3", branch3x3dbl_3);
}

torch::Tensor InceptionBImpl::forward(torch::Tensor x) {
  auto branch3x3 = this->branch3x3->forward(x);

  auto branch3x3dbl = this->branch3x3dbl_1->forward(x);
  branch3x3dbl = this->branch3x3dbl_2->forward(branch3x3dbl);
  branch3x3dbl = this->branch3x3dbl_3->forward(branch3x3dbl);

  auto branch_pool = torch::max_pool2d(x, 3, 2);
  return torch::cat({branch3x3, branch3x3dbl, branch_pool}, 1);
}

InceptionCImpl::InceptionCImpl(int64_t in_channels, int64_t channels_7x7) {
  branch1x1 = BasicConv2d(Options(in_channels, 192, 1));

  auto c7 = channels_7x7;
  branch7x7_1 = BasicConv2d(Options(in_channels, c7, 1));
  branch7x7_2 = BasicConv2d(Options(c7, c7, {1, 7}).padding({0, 3}));
  branch7x7_3 = BasicConv2d(Options(c7, 192, {7, 1}).padding({3, 0}));

  branch7x7dbl_1 = BasicConv2d(Options(in_channels, c7, 1));
  branch7x7dbl_2 = BasicConv2d(Options(c7, c7, {7, 1}).padding({3, 0}));
  branch7x7dbl_3 = BasicConv2d(Options(c7, c7, {1, 7}).padding({0, 3}));
  branch7x7dbl_4 = BasicConv2d(Options(c7, c7, {7, 1}).padding({3, 0}));
  branch7x7dbl_5 = BasicConv2d(Options(c7, 192, {1, 7}).padding({0, 3}));

  branch_pool = BasicConv2d(Options(in_channels, 192, 1));

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

torch::Tensor InceptionCImpl::forward(torch::Tensor x) {
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

InceptionDImpl::InceptionDImpl(int64_t in_channels)
    : branch3x3_1(Options(in_channels, 192, 1)),
      branch3x3_2(Options(192, 320, 3).stride(2)),
      branch7x7x3_1(Options(in_channels, 192, 1)),
      branch7x7x3_2(Options(192, 192, {1, 7}).padding({0, 3})),
      branch7x7x3_3(Options(192, 192, {7, 1}).padding({3, 0})),
      branch7x7x3_4(Options(192, 192, 3).stride(2))

{
  register_module("branch3x3_1", branch3x3_1);
  register_module("branch3x3_2", branch3x3_2);
  register_module("branch7x7x3_1", branch7x7x3_1);
  register_module("branch7x7x3_2", branch7x7x3_2);
  register_module("branch7x7x3_3", branch7x7x3_3);
  register_module("branch7x7x3_4", branch7x7x3_4);
}

torch::Tensor InceptionDImpl::forward(torch::Tensor x) {
  auto branch3x3 = this->branch3x3_1->forward(x);
  branch3x3 = this->branch3x3_2->forward(branch3x3);

  auto branch7x7x3 = this->branch7x7x3_1->forward(x);
  branch7x7x3 = this->branch7x7x3_2->forward(branch7x7x3);
  branch7x7x3 = this->branch7x7x3_3->forward(branch7x7x3);
  branch7x7x3 = this->branch7x7x3_4->forward(branch7x7x3);

  auto branch_pool = torch::max_pool2d(x, 3, 2);
  return torch::cat({branch3x3, branch7x7x3, branch_pool}, 1);
}

InceptionEImpl::InceptionEImpl(int64_t in_channels)
    : branch1x1(Options(in_channels, 320, 1)),
      branch3x3_1(Options(in_channels, 384, 1)),
      branch3x3_2a(Options(384, 384, {1, 3}).padding({0, 1})),
      branch3x3_2b(Options(384, 384, {3, 1}).padding({1, 0})),
      branch3x3dbl_1(Options(in_channels, 448, 1)),
      branch3x3dbl_2(Options(448, 384, 3).padding(1)),
      branch3x3dbl_3a(Options(384, 384, {1, 3}).padding({0, 1})),
      branch3x3dbl_3b(Options(384, 384, {3, 1}).padding({1, 0})),
      branch_pool(Options(in_channels, 192, 1)) {
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

torch::Tensor InceptionEImpl::forward(torch::Tensor x) {
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
  branch3x3dbl = torch::cat(
      {this->branch3x3dbl_3a->forward(branch3x3dbl),
       this->branch3x3dbl_3b->forward(branch3x3dbl)},
      1);

  auto branch_pool = torch::avg_pool2d(x, 3, 1, 1);
  branch_pool = this->branch_pool->forward(branch_pool);

  return torch::cat({branch1x1, branch3x3, branch3x3dbl, branch_pool}, 1);
}

InceptionAuxImpl::InceptionAuxImpl(int64_t in_channels, int64_t num_classes)
    : conv0(BasicConv2d(Options(in_channels, 128, 1))),
      conv1(BasicConv2d(Options(128, 768, 5), 0.01)),
      fc(768, num_classes) {
  torch::nn::init::normal_(
      fc->weight,
      0,
      0.001); // Note: used instead of truncated normal initialization

  register_module("conv0", conv0);
  register_module("conv1", conv1);
  register_module("fc", fc);
}

torch::Tensor InceptionAuxImpl::forward(torch::Tensor x) {
  // N x 768 x 17 x 17
  x = torch::avg_pool2d(x, 5, 3);
  // N x 768 x 5 x 5
  x = conv0->forward(x);
  // N x 128 x 5 x 5
  x = conv1->forward(x);
  // N x 768 x 1 x 1
  x = torch::adaptive_avg_pool2d(x, {1, 1});
  // N x 768 x 1 x 1
  x = x.view({x.size(0), -1});
  // N x 768
  x = fc->forward(x);
  // N x 1000 (num_classes)
  return x;
}

} // namespace _inceptionimpl

InceptionV3Impl::InceptionV3Impl(
    int64_t num_classes,
    bool aux_logits,
    bool transform_input)
    : aux_logits(aux_logits), transform_input(transform_input) {
  Conv2d_1a_3x3 = _inceptionimpl::BasicConv2d(Options(3, 32, 3).stride(2));
  Conv2d_2a_3x3 = _inceptionimpl::BasicConv2d(Options(32, 32, 3));
  Conv2d_2b_3x3 = _inceptionimpl::BasicConv2d(Options(32, 64, 3).padding(1));
  Conv2d_3b_1x1 = _inceptionimpl::BasicConv2d(Options(64, 80, 1));
  Conv2d_4a_3x3 = _inceptionimpl::BasicConv2d(Options(80, 192, 3));

  Mixed_5b = _inceptionimpl::InceptionA(192, 32);
  Mixed_5c = _inceptionimpl::InceptionA(256, 64);
  Mixed_5d = _inceptionimpl::InceptionA(288, 64);

  Mixed_6a = _inceptionimpl::InceptionB(288);
  Mixed_6b = _inceptionimpl::InceptionC(768, 128);
  Mixed_6c = _inceptionimpl::InceptionC(768, 160);
  Mixed_6d = _inceptionimpl::InceptionC(768, 160);
  Mixed_6e = _inceptionimpl::InceptionC(768, 192);

  if (aux_logits)
    AuxLogits = _inceptionimpl::InceptionAux(768, num_classes);

  Mixed_7a = _inceptionimpl::InceptionD(768);
  Mixed_7b = _inceptionimpl::InceptionE(1280);
  Mixed_7c = _inceptionimpl::InceptionE(2048);

  fc = torch::nn::Linear(2048, num_classes);
  torch::nn::init::normal_(
      fc->weight,
      0,
      0.1); // Note: used instead of truncated normal initialization

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

  if (!AuxLogits.is_empty())
    register_module("AuxLogits", AuxLogits);

  register_module("Mixed_7a", Mixed_7a);
  register_module("Mixed_7b", Mixed_7b);
  register_module("Mixed_7c", Mixed_7c);
  register_module("fc", fc);
}

InceptionV3Output InceptionV3Impl::forward(torch::Tensor x) {
  if (transform_input) {
    auto x_ch0 = torch::unsqueeze(x.select(1, 0), 1) * (0.229 / 0.5) +
        (0.485 - 0.5) / 0.5;
    auto x_ch1 = torch::unsqueeze(x.select(1, 1), 1) * (0.224 / 0.5) +
        (0.456 - 0.5) / 0.5;
    auto x_ch2 = torch::unsqueeze(x.select(1, 2), 1) * (0.225 / 0.5) +
        (0.406 - 0.5) / 0.5;

    x = torch::cat({x_ch0, x_ch1, x_ch2}, 1);
  }

  // N x 3 x 299 x 299
  x = Conv2d_1a_3x3->forward(x);
  // N x 32 x 149 x 149
  x = Conv2d_2a_3x3->forward(x);
  // N x 32 x 147 x 147
  x = Conv2d_2b_3x3->forward(x);
  // N x 64 x 147 x 147
  x = torch::max_pool2d(x, 3, 2);
  // N x 64 x 73 x 73
  x = Conv2d_3b_1x1->forward(x);
  // N x 80 x 73 x 73
  x = Conv2d_4a_3x3->forward(x);
  // N x 192 x 71 x 71
  x = torch::max_pool2d(x, 3, 2);
  // N x 192 x 35 x 35
  x = Mixed_5b->forward(x);
  // N x 256 x 35 x 35
  x = Mixed_5c->forward(x);
  // N x 288 x 35 x 35
  x = Mixed_5d->forward(x);
  // N x 288 x 35 x 35
  x = Mixed_6a->forward(x);
  // N x 768 x 17 x 17
  x = Mixed_6b->forward(x);
  // N x 768 x 17 x 17
  x = Mixed_6c->forward(x);
  // N x 768 x 17 x 17
  x = Mixed_6d->forward(x);
  // N x 768 x 17 x 17
  x = Mixed_6e->forward(x);
  // N x 768 x 17 x 17

  torch::Tensor aux;
  if (is_training() && aux_logits)
    aux = AuxLogits->forward(x);

  // N x 768 x 17 x 17
  x = Mixed_7a->forward(x);
  // N x 1280 x 8 x 8
  x = Mixed_7b->forward(x);
  // N x 2048 x 8 x 8
  x = Mixed_7c->forward(x);
  // N x 2048 x 8 x 8
  x = torch::adaptive_avg_pool2d(x, {1, 1});
  // N x 2048 x 1 x 1
  x = torch::dropout(x, 0.5, is_training());
  // N x 2048 x 1 x 1
  x = x.view({x.size(0), -1});
  // N x 2048
  x = fc->forward(x);
  // N x 1000 (num_classes)

  if (is_training() && aux_logits)
    return {x, aux};
  return {x, {}};
}

// namespace _inceptionimpl
} // namespace models
} // namespace vision
