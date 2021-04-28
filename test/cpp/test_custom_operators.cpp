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

TEST(test_custom_operators, roi_align_visible) {
  // make sure that the torchvision ops are visible to the jit interpreter even if
  // not explicitly included
  auto& ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString("torchvision::roi_align"));
  ASSERT_EQ(ops.size(), 1);

  auto& op = ops.front();
  ASSERT_EQ(op->schema().name(), "torchvision::roi_align");

  torch::jit::Stack stack;
  float roi_data[] = {
    0., 0., 0., 5., 5.,
    0., 5., 5., 10., 10.
  };
  at::Tensor input = at::rand({1, 2, 10, 10}), rois = at::from_blob(roi_data, {2, 5});
  double spatial_scale = 1.0;
  int64_t pooled_height = 3, pooled_width = 3, sampling_ratio = -1;
  bool aligned = true;

  torch::jit::push(stack, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
  op->getOperation()(&stack);
  at::Tensor output_jit;
  torch::jit::pop(stack, output_jit);

  ASSERT_EQ(output_jit.sizes()[0], 2);
  ASSERT_EQ(output_jit.sizes()[1], 2);
  ASSERT_EQ(output_jit.sizes()[2], 3);
  ASSERT_EQ(output_jit.sizes()[3], 3);
}
