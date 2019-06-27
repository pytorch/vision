#include "googlenet.h"

#include "modelsimpl.h"

namespace vision {
namespace models {

using Options = torch::nn::Conv2dOptions;

namespace _googlenetimpl {
BasicConv2dImpl::BasicConv2dImpl(torch::nn::Conv2dOptions options) {
  options.with_bias(false);
  conv = torch::nn::Conv2d(options);
  bn = torch::nn::BatchNorm(
      torch::nn::BatchNormOptions(options.output_channels()).eps(0.001));

  register_module("conv", conv);
  register_module("bn", bn);
}

torch::Tensor BasicConv2dImpl::forward(torch::Tensor x) {
  x = conv->forward(x);
  x = bn->forward(x);
  return x.relu_();
}

InceptionImpl::InceptionImpl(
    int64_t in_channels,
    int64_t ch1x1,
    int64_t ch3x3red,
    int64_t ch3x3,
    int64_t ch5x5red,
    int64_t ch5x5,
    int64_t pool_proj) {
  branch1 = BasicConv2d(Options(in_channels, ch1x1, 1));

  branch2->push_back(BasicConv2d(Options(in_channels, ch3x3red, 1)));
  branch2->push_back(BasicConv2d(Options(ch3x3red, ch3x3, 3).padding(1)));

  branch3->push_back(BasicConv2d(Options(in_channels, ch5x5red, 1)));
  branch3->push_back(BasicConv2d(Options(ch5x5red, ch5x5, 3).padding(1)));

  branch4->push_back(
      torch::nn::Functional(torch::max_pool2d, 3, 1, 1, 1, true));
  branch4->push_back(BasicConv2d(Options(in_channels, pool_proj, 1)));

  register_module("branch1", branch1);
  register_module("branch2", branch2);
  register_module("branch3", branch3);
  register_module("branch4", branch4);
}

torch::Tensor InceptionImpl::forward(torch::Tensor x) {
  auto b1 = branch1->forward(x);
  auto b2 = branch2->forward(x);
  auto b3 = branch3->forward(x);
  auto b4 = branch4->forward(x);

  return torch::cat({b1, b2, b3, b4}, 1);
}

InceptionAuxImpl::InceptionAuxImpl(int64_t in_channels, int64_t num_classes) {
  conv = BasicConv2d(Options(in_channels, 128, 1));
  fc1 = torch::nn::Linear(2048, 1024);
  fc2 = torch::nn::Linear(1024, num_classes);

  register_module("conv", conv);
  register_module("fc1", fc1);
  register_module("fc2", fc2);
}

torch::Tensor InceptionAuxImpl::forward(at::Tensor x) {
  // aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
  x = torch::adaptive_avg_pool2d(x, {4, 4});
  // aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
  x = conv->forward(x);
  // N x 128 x 4 x 4
  x = x.view({x.size(0), -1});
  // N x 2048
  x = fc1->forward(x).relu_();
  // N x 2048
  x = torch::dropout(x, 0.7, is_training());
  // N x 2048
  x = fc2->forward(x);
  // N x 1024

  return x;
}

} // namespace _googlenetimpl

GoogLeNetImpl::GoogLeNetImpl(
    int64_t num_classes,
    bool aux_logits,
    bool transform_input,
    bool init_weights) {
  this->aux_logits = aux_logits;
  this->transform_input = transform_input;

  conv1 = _googlenetimpl::BasicConv2d(Options(3, 64, 7).stride(2).padding(3));
  conv2 = _googlenetimpl::BasicConv2d(Options(64, 64, 1));
  conv3 = _googlenetimpl::BasicConv2d(Options(64, 192, 3).padding(1));

  inception3a = _googlenetimpl::Inception(192, 64, 96, 128, 16, 32, 32);
  inception3b = _googlenetimpl::Inception(256, 128, 128, 192, 32, 96, 64);

  inception4a = _googlenetimpl::Inception(480, 192, 96, 208, 16, 48, 64);
  inception4b = _googlenetimpl::Inception(512, 160, 112, 224, 24, 64, 64);
  inception4c = _googlenetimpl::Inception(512, 128, 128, 256, 24, 64, 64);
  inception4d = _googlenetimpl::Inception(512, 112, 144, 288, 32, 64, 64);
  inception4e = _googlenetimpl::Inception(528, 256, 160, 320, 32, 128, 128);

  inception5a = _googlenetimpl::Inception(832, 256, 160, 320, 32, 128, 128);
  inception5b = _googlenetimpl::Inception(832, 384, 192, 384, 48, 128, 128);

  if (aux_logits) {
    aux1 = _googlenetimpl::InceptionAux(512, num_classes);
    aux2 = _googlenetimpl::InceptionAux(528, num_classes);

    register_module("aux1", aux1);
    register_module("aux2", aux2);
  }

  dropout = torch::nn::Dropout(0.2);
  fc = torch::nn::Linear(1024, num_classes);

  register_module("conv1", conv1);
  register_module("conv2", conv2);
  register_module("conv3", conv3);

  register_module("inception3a", inception3a);
  register_module("inception3b", inception3b);

  register_module("inception4a", inception4a);
  register_module("inception4b", inception4b);
  register_module("inception4c", inception4c);
  register_module("inception4d", inception4d);
  register_module("inception4e", inception4e);

  register_module("inception5a", inception5a);
  register_module("inception5b", inception5b);

  register_module("dropout", dropout);
  register_module("fc", fc);

  if (init_weights)
    _initialize_weights();
}

void GoogLeNetImpl::_initialize_weights() {
  for (auto& module : modules(/*include_self=*/false)) {
    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
      torch::nn::init::normal_(M->weight); // Note: used instead of truncated
                                           // normal initialization
    else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get()))
      torch::nn::init::normal_(M->weight); // Note: used instead of truncated
                                           // normal initialization
    else if (auto M = dynamic_cast<torch::nn::BatchNormImpl*>(module.get())) {
      torch::nn::init::ones_(M->weight);
      torch::nn::init::zeros_(M->bias);
    }
  }
}

GoogLeNetOutput GoogLeNetImpl::forward(torch::Tensor x) {
  if (transform_input) {
    auto x_ch0 = torch::unsqueeze(x.select(1, 0), 1) * (0.229 / 0.5) +
        (0.485 - 0.5) / 0.5;
    auto x_ch1 = torch::unsqueeze(x.select(1, 1), 1) * (0.224 / 0.5) +
        (0.456 - 0.5) / 0.5;
    auto x_ch2 = torch::unsqueeze(x.select(1, 2), 1) * (0.225 / 0.5) +
        (0.406 - 0.5) / 0.5;

    x = torch::cat({x_ch0, x_ch1, x_ch2}, 1);
  }

  // N x 3 x 224 x 224
  x = conv1->forward(x);
  // N x 64 x 112 x 112
  x = torch::max_pool2d(x, 3, 2, 0, 1, true);
  // N x 64 x 56 x 56
  x = conv2->forward(x);
  // N x 64 x 56 x 56
  x = conv3->forward(x);
  // N x 192 x 56 x 56
  x = torch::max_pool2d(x, 3, 2, 0, 1, true);

  // N x 192 x 28 x 28
  x = inception3a->forward(x);
  // N x 256 x 28 x 28
  x = inception3b->forward(x);
  // N x 480 x 28 x 28
  x = torch::max_pool2d(x, 3, 2, 0, 1, true);
  // N x 480 x 14 x 14
  x = inception4a->forward(x);
  // N x 512 x 14 x 14
  torch::Tensor aux1;
  if (is_training() && aux_logits)
    aux1 = this->aux1->forward(x);

  x = inception4b->forward(x);
  // N x 512 x 14 x 14
  x = inception4c->forward(x);
  // N x 512 x 14 x 14
  x = inception4d->forward(x);
  // N x 528 x 14 x 14
  torch::Tensor aux2;
  if (is_training() && aux_logits)
    aux2 = this->aux2->forward(x);

  x = inception4e(x);
  // N x 832 x 14 x 14
  x = torch::max_pool2d(x, 2, 2, 0, 1, true);
  // N x 832 x 7 x 7
  x = inception5a(x);
  // N x 832 x 7 x 7
  x = inception5b(x);
  // N x 1024 x 7 x 7

  x = torch::adaptive_avg_pool2d(x, {1, 1});
  // N x 1024 x 1 x 1
  x = x.view({x.size(0), -1});
  // N x 1024
  x = dropout->forward(x);
  x = fc->forward(x);
  // N x 1000(num_classes)

  return {x, aux1, aux2};
}

} // namespace models
} // namespace vision
