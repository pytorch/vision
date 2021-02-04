// Copyright 2004-present Facebook. All Rights Reserved.

#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

// FIXME: the include path differs from OSS due to the extra csrc
#include <torchvision/csrc/ops/nms.h>

TEST(test_custom_operators, nms) {
  // make sure that the torchvision ops are visible to the jit interpreter
  auto& ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString("torchvision::nms"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "torchvision::nms");

  torch::jit::Stack stack;
  at::Tensor boxes = at::rand({50, 4}), scores = at::rand({50});
  double thresh = 0.7;

  torch::jit::push(stack, boxes, scores, thresh);
  op->getOperation()(&stack);
  at::Tensor output_jit;
  torch::jit::pop(stack, output_jit);

  at::Tensor output = vision::ops::nms(boxes, scores, thresh);
  ASSERT_TRUE(output_jit.allclose(output));

}
