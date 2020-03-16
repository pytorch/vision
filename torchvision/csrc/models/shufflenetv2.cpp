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
  opts = opts.stride(1).padding(0).bias(false);
  return torch::nn::Conv2d(opts);
}

torch::nn::Conv2d conv33(int64_t input, int64_t output, int64_t stride) {
  Options opts(input, output, 3);
  opts = opts.stride(stride).padding(1).bias(false).groups(input);
  return torch::nn::Conv2d(opts);
}

struct ShuffleNetV2InvertedResidualImpl : torch::nn::Module {
  int64_t stride;
  torch::nn::Sequential branch1{nullptr}, branch2{nullptr};

  ShuffleNetV2InvertedResidualImpl(int64_t inp, int64_t oup, int64_t stride)
      : stride(stride) {
    TORCH_CHECK(stride >= 1 && stride <= 3, "illegal stride value");

    auto branch_features = oup / 2;
    TORCH_CHECK(stride != 1 || inp == branch_features << 1);

    if (stride > 1) {
      branch1 = torch::nn::Sequential(
          conv33(inp, inp, stride),
          torch::nn::BatchNorm2d(inp),
          conv11(inp, branch_features),
          torch::nn::BatchNorm2d(branch_features),
          torch::nn::Functional(modelsimpl::relu_));
    }

    branch2 = torch::nn::Sequential(
        conv11(stride > 1 ? inp : branch_features, branch_features),
        torch::nn::BatchNorm2d(branch_features),
        torch::nn::Functional(modelsimpl::relu_),
        conv33(branch_features, branch_features, stride),
        torch::nn::BatchNorm2d(branch_features),
        conv11(branch_features, branch_features),
        torch::nn::BatchNorm2d(branch_features),
        torch::nn::Functional(modelsimpl::relu_));

    if (!branch1.is_empty())
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

    out = channel_shuffle(out, 2);
    return out;
  }
};

TORCH_MODULE(ShuffleNetV2InvertedResidual);

ShuffleNetV2Impl::ShuffleNetV2Impl(
    const std::vector<int64_t>& stage_repeats,
    const std::vector<int64_t>& stage_out_channels,
    int64_t num_classes) {
  TORCH_CHECK(
      stage_repeats.size() == 3,
      "expected stage_repeats as vector of 3 positive ints");

  TORCH_CHECK(
      stage_out_channels.size() == 5,
      "expected stage_out_channels as vector of 5 positive ints");

  _stage_out_channels = stage_out_channels;
  int64_t input_channels = 3;
  auto output_channels = _stage_out_channels[0];

  conv1 = torch::nn::Sequential(
      torch::nn::Conv2d(Options(input_channels, output_channels, 3)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
      torch::nn::BatchNorm2d(output_channels),
      torch::nn::Functional(modelsimpl::relu_));

  input_channels = output_channels;
  std::vector<torch::nn::Sequential> stages = {stage2, stage3, stage4};

  for (size_t i = 0; i < stages.size(); ++i) {
    auto& seq = stages[i];
    auto repeats = stage_repeats[i];
    auto output_channels = _stage_out_channels[i + 1];

    seq->push_back(
        ShuffleNetV2InvertedResidual(input_channels, output_channels, 2));

    for (size_t j = 0; j < size_t(repeats - 1); ++j)
      seq->push_back(
          ShuffleNetV2InvertedResidual(output_channels, output_channels, 1));

    input_channels = output_channels;
  }

  output_channels = _stage_out_channels.back();
  conv5 = torch::nn::Sequential(
      torch::nn::Conv2d(Options(input_channels, output_channels, 1)
                            .stride(1)
                            .padding(0)
                            .bias(false)),
      torch::nn::BatchNorm2d(output_channels),
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

ShuffleNetV2_x0_5Impl::ShuffleNetV2_x0_5Impl(int64_t num_classes)
    : ShuffleNetV2Impl({4, 8, 4}, {24, 48, 96, 192, 1024}, num_classes) {}

ShuffleNetV2_x1_0Impl::ShuffleNetV2_x1_0Impl(int64_t num_classes)
    : ShuffleNetV2Impl({4, 8, 4}, {24, 116, 232, 464, 1024}, num_classes) {}

ShuffleNetV2_x1_5Impl::ShuffleNetV2_x1_5Impl(int64_t num_classes)
    : ShuffleNetV2Impl({4, 8, 4}, {24, 176, 352, 704, 1024}, num_classes) {}

ShuffleNetV2_x2_0Impl::ShuffleNetV2_x2_0Impl(int64_t num_classes)
    : ShuffleNetV2Impl({4, 8, 4}, {24, 244, 488, 976, 2048}, num_classes) {}

} // namespace models
} // namespace vision
