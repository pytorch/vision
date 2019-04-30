#include "shufflenetv2.h"

#include "modelsimpl.h"

namespace vision {
namespace models {
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
  torch::nn::Conv2dOptions opts(input, output, 1);
  opts = opts.stride(1).padding(0).with_bias(false);
  return torch::nn::Conv2d(opts);
}

torch::nn::Conv2d conv33(int64_t input, int64_t output, int64_t stride) {
  torch::nn::Conv2dOptions opts(input, output, 3);
  opts = opts.stride(stride).padding(1).with_bias(false).groups(input);
  return torch::nn::Conv2d(opts);
}

struct InvertedResidual : torch::nn::Module {
  int64_t stride;
  torch::nn::Sequential branch1{nullptr}, branch2{nullptr};

  InvertedResidual(int64_t inp, int64_t oup, int64_t stride) : stride(stride) {
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

} // namespace models
} // namespace vision
