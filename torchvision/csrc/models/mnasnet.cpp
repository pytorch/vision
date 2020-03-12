#include "mnasnet.h"

#include "modelsimpl.h"

namespace vision {
namespace models {
using Options = torch::nn::Conv2dOptions;

struct MNASNetInvertedResidualImpl : torch::nn::Module {
  bool apply_residual;
  torch::nn::Sequential layers;

  MNASNetInvertedResidualImpl(
      int64_t input,
      int64_t output,
      int64_t kernel,
      int64_t stride,
      double expansion_factor,
      double bn_momentum = 0.1) {
    TORCH_CHECK(stride == 1 || stride == 2);
    TORCH_CHECK(kernel == 3 || kernel == 5);

    auto mid = int64_t(input * expansion_factor);
    apply_residual = input == output && stride == 1;

    layers->push_back(torch::nn::Conv2d(Options(input, mid, 1).bias(false)));
    layers->push_back(torch::nn::BatchNorm2d(
        torch::nn::BatchNormOptions(mid).momentum(bn_momentum)));
    layers->push_back(
        torch::nn::Functional(torch::nn::Functional(modelsimpl::relu_)));
    layers->push_back(
        torch::nn::Conv2d(torch::nn::Conv2d(Options(mid, mid, kernel)
                                                .padding(kernel / 2)
                                                .stride(stride)
                                                .groups(mid)
                                                .bias(false))));
    layers->push_back(torch::nn::BatchNorm2d(
        torch::nn::BatchNormOptions(mid).momentum(bn_momentum)));
    layers->push_back(
        torch::nn::Functional(torch::nn::Functional(modelsimpl::relu_)));
    layers->push_back(torch::nn::Conv2d(Options(mid, output, 1).bias(false)));
    layers->push_back(torch::nn::BatchNorm2d(
        torch::nn::BatchNormOptions(output).momentum(bn_momentum)));

    register_module("layers", layers);
  }

  torch::Tensor forward(torch::Tensor x) {
    if (apply_residual)
      return layers->forward(x) + x;
    return layers->forward(x);
  }
};

TORCH_MODULE(MNASNetInvertedResidual);

struct StackSequentailImpl : torch::nn::SequentialImpl {
  using SequentialImpl::SequentialImpl;

  torch::Tensor forward(torch::Tensor x) {
    return SequentialImpl::forward(x);
  }
};

TORCH_MODULE(StackSequentail);

StackSequentail stack(
    int64_t input,
    int64_t output,
    int64_t kernel,
    int64_t stride,
    double exp_factor,
    int64_t repeats,
    double bn_momentum) {
  TORCH_CHECK(repeats >= 1);

  StackSequentail seq;
  seq->push_back(MNASNetInvertedResidual(
      input, output, kernel, stride, exp_factor, bn_momentum));

  for (int64_t i = 1; i < repeats; ++i)
    seq->push_back(MNASNetInvertedResidual(
        output, output, kernel, 1, exp_factor, bn_momentum));

  return seq;
}

int64_t round_to_multiple_of(
    int64_t val,
    int64_t divisor,
    double round_up_bias = .9) {
  TORCH_CHECK(0.0 < round_up_bias && round_up_bias < 1.0);
  auto new_val = std::max(divisor, (val + divisor / 2) / divisor * divisor);
  return new_val >= round_up_bias * val ? new_val : new_val + divisor;
}

std::vector<int64_t> scale_depths(std::vector<int64_t> depths, double alpha) {
  std::vector<int64_t> data(depths.size());
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = round_to_multiple_of(int64_t(depths[i] * alpha), 8);
  }

  return data;
}

void MNASNetImpl::_initialize_weights() {
  for (auto& module : modules(/*include_self=*/false)) {
    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
      torch::nn::init::kaiming_normal_(
          M->weight, 0, torch::kFanOut, torch::kReLU);
    else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
      torch::nn::init::ones_(M->weight);
      torch::nn::init::zeros_(M->bias);
    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
      torch::nn::init::normal_(M->weight, 0, 0.01);
      torch::nn::init::zeros_(M->bias);
    }
  }
}

#define BN_MOMENTUM 1 - 0.9997

MNASNetImpl::MNASNetImpl(double alpha, int64_t num_classes, double dropout) {
  auto depths = scale_depths({24, 40, 80, 96, 192, 320}, alpha);

  layers->push_back(
      torch::nn::Conv2d(Options(3, 32, 3).padding(1).stride(2).bias(false)));
  layers->push_back(torch::nn::BatchNorm2d(
      torch::nn::BatchNormOptions(32).momentum(BN_MOMENTUM)));
  layers->push_back(torch::nn::Functional(modelsimpl::relu_));
  layers->push_back(torch::nn::Conv2d(
      Options(32, 32, 3).padding(1).stride(1).groups(32).bias(false)));
  layers->push_back(torch::nn::BatchNorm2d(
      torch::nn::BatchNormOptions(32).momentum(BN_MOMENTUM)));
  layers->push_back(torch::nn::Functional(modelsimpl::relu_));
  layers->push_back(
      torch::nn::Conv2d(Options(32, 16, 1).padding(0).stride(1).bias(false)));
  layers->push_back(torch::nn::BatchNorm2d(
      torch::nn::BatchNormOptions(16).momentum(BN_MOMENTUM)));

  layers->push_back(stack(16, depths[0], 3, 2, 3, 3, BN_MOMENTUM));
  layers->push_back(stack(depths[0], depths[1], 5, 2, 3, 3, BN_MOMENTUM));
  layers->push_back(stack(depths[1], depths[2], 5, 2, 6, 3, BN_MOMENTUM));
  layers->push_back(stack(depths[2], depths[3], 3, 1, 6, 2, BN_MOMENTUM));
  layers->push_back(stack(depths[3], depths[4], 5, 2, 6, 4, BN_MOMENTUM));
  layers->push_back(stack(depths[4], depths[5], 3, 1, 6, 1, BN_MOMENTUM));

  layers->push_back(torch::nn::Conv2d(
      Options(depths[5], 1280, 1).padding(0).stride(1).bias(false)));
  layers->push_back(torch::nn::BatchNorm2d(
      torch::nn::BatchNormOptions(1280).momentum(BN_MOMENTUM)));
  layers->push_back(torch::nn::Functional(modelsimpl::relu_));

  classifier = torch::nn::Sequential(
      torch::nn::Dropout(dropout), torch::nn::Linear(1280, num_classes));

  register_module("layers", layers);
  register_module("classifier", classifier);

  _initialize_weights();
}

torch::Tensor MNASNetImpl::forward(torch::Tensor x) {
  x = layers->forward(x);
  x = x.mean({2, 3});
  return classifier->forward(x);
}

MNASNet0_5Impl::MNASNet0_5Impl(int64_t num_classes, double dropout)
    : MNASNetImpl(.5, num_classes, dropout) {}

MNASNet0_75Impl::MNASNet0_75Impl(int64_t num_classes, double dropout)
    : MNASNetImpl(.75, num_classes, dropout) {}

MNASNet1_0Impl::MNASNet1_0Impl(int64_t num_classes, double dropout)
    : MNASNetImpl(1, num_classes, dropout) {}

MNASNet1_3Impl::MNASNet1_3Impl(int64_t num_classes, double dropout)
    : MNASNetImpl(1.3, num_classes, dropout) {}

} // namespace models
} // namespace vision
