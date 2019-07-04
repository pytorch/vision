#ifndef INCEPTION_H
#define INCEPTION_H

#include <torch/torch.h>
#include "general.h"

namespace vision {
namespace models {
namespace _inceptionimpl {
struct VISION_API BasicConv2dImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm bn{nullptr};

  BasicConv2dImpl(torch::nn::Conv2dOptions options, double std_dev = 0.1);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BasicConv2d);

struct VISION_API InceptionAImpl : torch::nn::Module {
  BasicConv2d branch1x1, branch5x5_1, branch5x5_2, branch3x3dbl_1,
      branch3x3dbl_2, branch3x3dbl_3, branch_pool;

  InceptionAImpl(int64_t in_channels, int64_t pool_features);

  torch::Tensor forward(torch::Tensor x);
};

struct VISION_API InceptionBImpl : torch::nn::Module {
  BasicConv2d branch3x3, branch3x3dbl_1, branch3x3dbl_2, branch3x3dbl_3;

  InceptionBImpl(int64_t in_channels);

  torch::Tensor forward(torch::Tensor x);
};

struct VISION_API InceptionCImpl : torch::nn::Module {
  BasicConv2d branch1x1{nullptr}, branch7x7_1{nullptr}, branch7x7_2{nullptr},
      branch7x7_3{nullptr}, branch7x7dbl_1{nullptr}, branch7x7dbl_2{nullptr},
      branch7x7dbl_3{nullptr}, branch7x7dbl_4{nullptr}, branch7x7dbl_5{nullptr},
      branch_pool{nullptr};

  InceptionCImpl(int64_t in_channels, int64_t channels_7x7);

  torch::Tensor forward(torch::Tensor x);
};

struct VISION_API InceptionDImpl : torch::nn::Module {
  BasicConv2d branch3x3_1, branch3x3_2, branch7x7x3_1, branch7x7x3_2,
      branch7x7x3_3, branch7x7x3_4;

  InceptionDImpl(int64_t in_channels);

  torch::Tensor forward(torch::Tensor x);
};

struct VISION_API InceptionEImpl : torch::nn::Module {
  BasicConv2d branch1x1, branch3x3_1, branch3x3_2a, branch3x3_2b,
      branch3x3dbl_1, branch3x3dbl_2, branch3x3dbl_3a, branch3x3dbl_3b,
      branch_pool;

  InceptionEImpl(int64_t in_channels);

  torch::Tensor forward(torch::Tensor x);
};

struct VISION_API InceptionAuxImpl : torch::nn::Module {
  BasicConv2d conv0;
  BasicConv2d conv1;
  torch::nn::Linear fc;

  InceptionAuxImpl(int64_t in_channels, int64_t num_classes);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(InceptionA);
TORCH_MODULE(InceptionB);
TORCH_MODULE(InceptionC);
TORCH_MODULE(InceptionD);
TORCH_MODULE(InceptionE);
TORCH_MODULE(InceptionAux);

} // namespace _inceptionimpl

struct VISION_API InceptionV3Output {
  torch::Tensor output;
  torch::Tensor aux;
};

// Inception v3 model architecture from
//"Rethinking the Inception Architecture for Computer Vision"
//<http://arxiv.org/abs/1512.00567>
struct VISION_API InceptionV3Impl : torch::nn::Module {
  bool aux_logits, transform_input;

  _inceptionimpl::BasicConv2d Conv2d_1a_3x3{nullptr}, Conv2d_2a_3x3{nullptr},
      Conv2d_2b_3x3{nullptr}, Conv2d_3b_1x1{nullptr}, Conv2d_4a_3x3{nullptr};

  _inceptionimpl::InceptionA Mixed_5b{nullptr}, Mixed_5c{nullptr},
      Mixed_5d{nullptr};
  _inceptionimpl::InceptionB Mixed_6a{nullptr};
  _inceptionimpl::InceptionC Mixed_6b{nullptr}, Mixed_6c{nullptr},
      Mixed_6d{nullptr}, Mixed_6e{nullptr};
  _inceptionimpl::InceptionD Mixed_7a{nullptr};
  _inceptionimpl::InceptionE Mixed_7b{nullptr}, Mixed_7c{nullptr};

  torch::nn::Linear fc{nullptr};

  _inceptionimpl::InceptionAux AuxLogits{nullptr};

  InceptionV3Impl(
      int64_t num_classes = 1000,
      bool aux_logits = true,
      bool transform_input = false);

  InceptionV3Output forward(torch::Tensor x);
};

TORCH_MODULE(InceptionV3);

} // namespace models
} // namespace vision

#endif // INCEPTION_H
