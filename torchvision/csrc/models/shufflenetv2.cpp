#include "shufflenetv2.h"

#include "modelsimpl.h"

namespace vision {
namespace models {

using Options = torch::nn::Conv2dOptions;

torch::Tensor channel_shuffle(torch::Tensor x, int64_t groups) {
  auto shape = x.sizes();
  auto batchsize = shape[0];
  auto num_channels = shape[1];
  auto height = shape[2];
  auto width = shape[3];

  auto channels_per_group = num_channels / groups;

  x = x.view({batchsize, groups, channels_per_group, height, width});
  x = torch::transpose(x, 1, 2).contiguous();
  x = x.view({batchsize, -1, height, width});

  return x;
}

torch::nn::Conv2d conv11(int64_t input, int64_t output) {
  Options opts(input, output, 1);
  opts = opts.stride(1).padding(0).with_bias(false);
  return torch::nn::Conv2d(opts);
}

torch::nn::Conv2d conv33(int64_t input, int64_t output, int64_t stride) {
  Options opts(input, output, 3);
  opts = opts.stride(stride).padding(1).with_bias(false).groups(input);
  return torch::nn::Conv2d(opts);
}

struct InvertedResidualImpl : torch::nn::Module {
  int64_t stride;
  torch::nn::Sequential branch1{nullptr}, branch2{nullptr};

  InvertedResidualImpl(int64_t inp, int64_t oup, int64_t stride)
      : stride(stride) {
    if (stride < 1 || stride > 3) {
      std::cerr << "illegal stride value'" << std::endl;
      assert(false);
    }

    auto branch_features = oup / 2;
    assert(stride != 1 || inp == branch_features << 1);

    if (stride > 1) {
      branch1 = torch::nn::Sequential(
          conv33(inp, inp, stride),
          torch::nn::BatchNorm(inp),
          conv11(inp, branch_features),
          torch::nn::BatchNorm(branch_features),
          torch::nn::Functional(modelsimpl::relu_));
    }

    branch2 = torch::nn::Sequential(
        conv11(stride > 1 ? inp : branch_features, branch_features),
        torch::nn::BatchNorm(branch_features),
        torch::nn::Functional(modelsimpl::relu_),
        conv33(branch_features, branch_features, stride),
        torch::nn::BatchNorm(branch_features),
        conv11(branch_features, branch_features),
        torch::nn::BatchNorm(branch_features),
        torch::nn::Functional(modelsimpl::relu_));

    if (branch1)
      register_module("branch1", branch1);

    register_module("branch2", branch2);
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor out;

    if (stride == 1) {
      auto chunks = x.chunk(2, 1);
      out = torch::cat({chunks[0], branch2->forward(chunks[1])}, 1);
    } else
      out = torch::cat({branch1->forward(x), branch2->forward(x)}, 1);

    out = channel_shuffle(x, 2);
    return out;
  }
};

TORCH_MODULE(InvertedResidual);

static std::unordered_map<double, std::vector<int64_t>> channels = {
    {0.5, {24, 48, 96, 192, 1024}},
    {1.0, {24, 116, 232, 464, 1024}},
    {1.5, {24, 176, 352, 704, 1024}},
    {2.0, {24, 244, 488, 976, 2048}}};

std::vector<int64_t> ShuffleNetV2Impl::_get_stages(double mult) {
  for (const auto& P : channels)
    if (modelsimpl::double_compare(mult, P.first))
      return P.second;

  std::cerr << "width_mult" << mult << "is not supported" << std::endl;
  assert(false);
}

ShuffleNetV2Impl::ShuffleNetV2Impl(
    int64_t num_classes,
    int64_t input_size,
    double width_mult) {
  stage_out_channels = _get_stages(width_mult);

  int64_t input_channels = 3;
  auto output_channels = stage_out_channels[0];

  conv1 = torch::nn::Sequential(
      torch::nn::Conv2d(Options(input_channels, output_channels, 3)
                            .stride(2)
                            .padding(1)
                            .with_bias(false)),
      torch::nn::BatchNorm(output_channels),
      torch::nn::Functional(modelsimpl::relu_));

  input_channels = output_channels;
  std::vector<int64_t> stage_repeats = {4, 8, 4};
  std::vector<torch::nn::Sequential*> stages = {&stage2, &stage3, &stage4};

  for (size_t i = 0; i < stages.size(); ++i) {
    auto seq = stages[i];
    auto repeats = stage_repeats[i];
    auto output_channels = stage_out_channels[i + 1];

    *seq = torch::nn::Sequential(
        InvertedResidual(input_channels, output_channels, 2));

    for (size_t j = 0; j < size_t(repeats - 1); ++j)
      seq->get()->push_back(
          InvertedResidual(output_channels, output_channels, 1));

    input_channels = output_channels;
  }

  output_channels = stage_out_channels.back();
  conv5 = torch::nn::Sequential(
      torch::nn::Conv2d(Options(input_channels, output_channels, 1)
                            .stride(1)
                            .padding(0)
                            .with_bias(false)),
      torch::nn::BatchNorm(output_channels),
      torch::nn::Functional(modelsimpl::relu_));

  fc = torch::nn::Linear(output_channels, num_classes);

  register_module("conv1", conv1);
  register_module("stage2", stage2);
  register_module("stage3", stage3);
  register_module("stage4", stage4);
  register_module("conv2", conv5);
  register_module("fc", fc);
}

torch::Tensor ShuffleNetV2Impl::forward(torch::Tensor x) {
  x = conv1->forward(x);
  x = torch::max_pool2d(x, 3, 2, 1);

  x = stage2->forward(x);
  x = stage3->forward(x);
  x = stage4->forward(x);
  x = conv5->forward(x);

  x = x.mean({2, 3});
  x = fc->forward(x);
  return x;
}

ShuffleNetV2_x0_5Impl::ShuffleNetV2_x0_5Impl(
    int64_t num_classes,
    int64_t input_size,
    double width_mult)
    : ShuffleNetV2Impl(num_classes, input_size, width_mult) {}

ShuffleNetV2_x1_0Impl::ShuffleNetV2_x1_0Impl(
    int64_t num_classes,
    int64_t input_size,
    double width_mult)
    : ShuffleNetV2Impl(num_classes, input_size, width_mult) {}

ShuffleNetV2_x1_5Impl::ShuffleNetV2_x1_5Impl(
    int64_t num_classes,
    int64_t input_size,
    double width_mult)
    : ShuffleNetV2Impl(num_classes, input_size, width_mult) {}

ShuffleNetV2_x2_0Impl::ShuffleNetV2_x2_0Impl(
    int64_t num_classes,
    int64_t input_size,
    double width_mult)
    : ShuffleNetV2Impl(num_classes, input_size, width_mult) {}

} // namespace models
} // namespace vision
