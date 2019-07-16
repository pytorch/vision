#include "densenet.h"

#include "modelsimpl.h"

namespace vision {
namespace models {
using Options = torch::nn::Conv2dOptions;

struct _DenseLayerImpl : torch::nn::SequentialImpl {
  double drop_rate;

  _DenseLayerImpl(
      int64_t num_input_features,
      int64_t growth_rate,
      int64_t bn_size,
      double drop_rate)
      : drop_rate(drop_rate) {
    push_back("norm1", torch::nn::BatchNorm(num_input_features));
    push_back("relu1", torch::nn::Functional(modelsimpl::relu_));
    push_back(
        "conv1",
        torch::nn::Conv2d(Options(num_input_features, bn_size * growth_rate, 1)
                              .stride(1)
                              .with_bias(false)));
    push_back("norm2", torch::nn::BatchNorm(bn_size * growth_rate));
    push_back("relu2", torch::nn::Functional(modelsimpl::relu_));
    push_back(
        "conv2",
        torch::nn::Conv2d(Options(bn_size * growth_rate, growth_rate, 3)
                              .stride(1)
                              .padding(1)
                              .with_bias(false)));
  }

  torch::Tensor forward(torch::Tensor x) {
    auto new_features = torch::nn::SequentialImpl::forward(x);
    if (drop_rate > 0)
      new_features =
          torch::dropout(new_features, drop_rate, this->is_training());
    return torch::cat({x, new_features}, 1);
  }
};

TORCH_MODULE(_DenseLayer);

struct _DenseBlockImpl : torch::nn::SequentialImpl {
  _DenseBlockImpl(
      int64_t num_layers,
      int64_t num_input_features,
      int64_t bn_size,
      int64_t growth_rate,
      double drop_rate) {
    for (int64_t i = 0; i < num_layers; ++i) {
      auto layer = _DenseLayer(
          num_input_features + i * growth_rate,
          growth_rate,
          bn_size,
          drop_rate);
      push_back("denselayer" + std::to_string(i + 1), layer);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    return torch::nn::SequentialImpl::forward(x);
  }
};

TORCH_MODULE(_DenseBlock);

struct _TransitionImpl : torch::nn::SequentialImpl {
  _TransitionImpl(int64_t num_input_features, int64_t num_output_features) {
    push_back("norm", torch::nn::BatchNorm(num_input_features));
    push_back("relu ", torch::nn::Functional(modelsimpl::relu_));
    push_back(
        "conv",
        torch::nn::Conv2d(Options(num_input_features, num_output_features, 1)
                              .stride(1)
                              .with_bias(false)));
    push_back(
        "pool", torch::nn::Functional([](torch::Tensor input) { return torch::avg_pool2d(input, 2, 2, 0, false, true); }));
  }

  torch::Tensor forward(torch::Tensor x) {
    return torch::nn::SequentialImpl::forward(x);
  }
};

TORCH_MODULE(_Transition);

DenseNetImpl::DenseNetImpl(
    int64_t num_classes,
    int64_t growth_rate,
    std::vector<int64_t> block_config,
    int64_t num_init_features,
    int64_t bn_size,
    double drop_rate) {
  // First convolution
  features = torch::nn::Sequential();
  features->push_back(
      "conv0",
      torch::nn::Conv2d(Options(3, num_init_features, 7)
                            .stride(2)
                            .padding(3)
                            .with_bias(false)));

  features->push_back("norm0", torch::nn::BatchNorm(num_init_features));
  features->push_back("relu0", torch::nn::Functional(modelsimpl::relu_));
  features->push_back(
      "pool0", torch::nn::Functional(torch::max_pool2d, 3, 2, 1, 1, false));

  // Each denseblock
  auto num_features = num_init_features;
  for (size_t i = 0; i < block_config.size(); ++i) {
    auto num_layers = block_config[i];
    _DenseBlock block(
        num_layers, num_features, bn_size, growth_rate, drop_rate);

    features->push_back("denseblock" + std::to_string(i + 1), block);
    num_features = num_features + num_layers * growth_rate;

    if (i != block_config.size() - 1) {
      auto trans = _Transition(num_features, num_features / 2);
      features->push_back("transition" + std::to_string(i + 1), trans);
      num_features = num_features / 2;
    }
  }

  // Final batch norm
  features->push_back("norm5", torch::nn::BatchNorm(num_features));
  // Linear layer
  classifier = torch::nn::Linear(num_features, num_classes);

  register_module("features", features);
  register_module("classifier", classifier);

  // Official init from torch repo.
  for (auto& module : modules(/*include_self=*/false)) {
    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
      torch::nn::init::kaiming_normal_(M->weight);
    else if (auto M = dynamic_cast<torch::nn::BatchNormImpl*>(module.get())) {
      torch::nn::init::constant_(M->weight, 1);
      torch::nn::init::constant_(M->bias, 0);
    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get()))
      torch::nn::init::constant_(M->bias, 0);
  }
}

torch::Tensor DenseNetImpl::forward(torch::Tensor x) {
  auto features = this->features->forward(x);
  auto out = torch::relu_(features);
  out = torch::adaptive_avg_pool2d(out, {1, 1});

  out = out.view({features.size(0), -1});
  out = this->classifier->forward(out);
  return out;
}

DenseNet121Impl::DenseNet121Impl(
    int64_t num_classes,
    int64_t growth_rate,
    std::vector<int64_t> block_config,
    int64_t num_init_features,
    int64_t bn_size,
    double drop_rate)
    : DenseNetImpl(
          num_classes,
          growth_rate,
          block_config,
          num_init_features,
          bn_size,
          drop_rate) {}

DenseNet169Impl::DenseNet169Impl(
    int64_t num_classes,
    int64_t growth_rate,
    std::vector<int64_t> block_config,
    int64_t num_init_features,
    int64_t bn_size,
    double drop_rate)
    : DenseNetImpl(
          num_classes,
          growth_rate,
          block_config,
          num_init_features,
          bn_size,
          drop_rate) {}

DenseNet201Impl::DenseNet201Impl(
    int64_t num_classes,
    int64_t growth_rate,
    std::vector<int64_t> block_config,
    int64_t num_init_features,
    int64_t bn_size,
    double drop_rate)
    : DenseNetImpl(
          num_classes,
          growth_rate,
          block_config,
          num_init_features,
          bn_size,
          drop_rate) {}

DenseNet161Impl::DenseNet161Impl(
    int64_t num_classes,
    int64_t growth_rate,
    std::vector<int64_t> block_config,
    int64_t num_init_features,
    int64_t bn_size,
    double drop_rate)
    : DenseNetImpl(
          num_classes,
          growth_rate,
          block_config,
          num_init_features,
          bn_size,
          drop_rate) {}

} // namespace models
} // namespace vision
