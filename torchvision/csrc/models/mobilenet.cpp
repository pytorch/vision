#include "mobilenet.h"

#include "modelsimpl.h"

namespace vision {
namespace models {
using Options = torch::nn::Conv2dOptions;

int64_t make_divisible(
    double value,
    int64_t divisor,
    c10::optional<int64_t> min_value = {}) {
  if (!min_value.has_value())
    min_value = divisor;
  auto new_value = std::max(
      min_value.value(), (int64_t(value + divisor / 2) / divisor) * divisor);
  if (new_value < .9 * value)
    new_value += divisor;
  return new_value;
}

struct ConvBNReLUImpl : torch::nn::SequentialImpl {
  ConvBNReLUImpl(
      int64_t in_planes,
      int64_t out_planes,
      int64_t kernel_size = 3,
      int64_t stride = 1,
      int64_t groups = 1) {
    auto padding = (kernel_size - 1) / 2;

    push_back(torch::nn::Conv2d(Options(in_planes, out_planes, kernel_size)
                                    .stride(stride)
                                    .padding(padding)
                                    .groups(groups)
                                    .bias(false)));
    push_back(torch::nn::BatchNorm(out_planes));
    push_back(torch::nn::Functional(modelsimpl::relu6_));
  }

  torch::Tensor forward(torch::Tensor x) {
    return torch::nn::SequentialImpl::forward(x);
  }
};

TORCH_MODULE(ConvBNReLU);

struct MobileNetInvertedResidualImpl : torch::nn::Module {
  int64_t stride;
  bool use_res_connect;
  torch::nn::Sequential conv;

  MobileNetInvertedResidualImpl(
      int64_t input,
      int64_t output,
      int64_t stride,
      double expand_ratio)
      : stride(stride), use_res_connect(stride == 1 && input == output) {
    auto double_compare = [](double a, double b) {
      return double(std::abs(a - b)) < std::numeric_limits<double>::epsilon();
    };

    TORCH_CHECK(stride == 1 || stride == 2);
    auto hidden_dim = int64_t(std::round(input * expand_ratio));

    if (!double_compare(expand_ratio, 1))
      conv->push_back(ConvBNReLU(input, hidden_dim, 1));

    conv->push_back(ConvBNReLU(hidden_dim, hidden_dim, 3, stride, hidden_dim));
    conv->push_back(torch::nn::Conv2d(
        Options(hidden_dim, output, 1).stride(1).padding(0).bias(false)));
    conv->push_back(torch::nn::BatchNorm(output));

    register_module("conv", conv);
  }

  torch::Tensor forward(torch::Tensor x) {
    if (use_res_connect)
      return x + conv->forward(x);
    return conv->forward(x);
  }
};

TORCH_MODULE(MobileNetInvertedResidual);

MobileNetV2Impl::MobileNetV2Impl(
    int64_t num_classes,
    double width_mult,
    std::vector<std::vector<int64_t>> inverted_residual_settings,
    int64_t round_nearest) {
  using Block = MobileNetInvertedResidual;
  int64_t input_channel = 32;
  int64_t last_channel = 1280;

  if (inverted_residual_settings.empty())
    inverted_residual_settings = {
        // t, c, n, s
        {1, 16, 1, 1},
        {6, 24, 2, 2},
        {6, 32, 3, 2},
        {6, 64, 4, 2},
        {6, 96, 3, 1},
        {6, 160, 3, 2},
        {6, 320, 1, 1},
    };

  TORCH_CHECK(
      inverted_residual_settings[0].size() == 4,
      "inverted_residual_settings should contain 4-element vectors");

  input_channel = make_divisible(input_channel * width_mult, round_nearest);
  this->last_channel =
      make_divisible(last_channel * std::max(1.0, width_mult), round_nearest);
  features->push_back(ConvBNReLU(3, input_channel, 3, 2));

  for (auto setting : inverted_residual_settings) {
    auto output_channel =
        make_divisible(setting[1] * width_mult, round_nearest);

    for (int64_t i = 0; i < setting[2]; ++i) {
      auto stride = i == 0 ? setting[3] : 1;
      features->push_back(
          Block(input_channel, output_channel, stride, setting[0]));
      input_channel = output_channel;
    }
  }

  features->push_back(ConvBNReLU(input_channel, this->last_channel, 1));

  classifier->push_back(torch::nn::Dropout(0.2));
  classifier->push_back(torch::nn::Linear(this->last_channel, num_classes));

  register_module("features", features);
  register_module("classifier", classifier);

  for (auto& module : modules(/*include_self=*/false)) {
    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
      torch::nn::init::kaiming_normal_(
          M->weight, 0, torch::nn::init::FanMode::FanOut);
      if (M->options.bias())
        torch::nn::init::zeros_(M->bias);
    } else if (auto M = dynamic_cast<torch::nn::BatchNormImpl*>(module.get())) {
      torch::nn::init::ones_(M->weight);
      torch::nn::init::zeros_(M->bias);
    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
      torch::nn::init::normal_(M->weight, 0, 0.01);
      torch::nn::init::zeros_(M->bias);
    }
  }
}

torch::Tensor MobileNetV2Impl::forward(at::Tensor x) {
  x = features->forward(x);
  x = x.mean({2, 3});
  x = classifier->forward(x);
  return x;
}

} // namespace models
} // namespace vision
