#include <torch/script.h>

#include "ROIAlign.h"
#include "ROIPool.h"
#include "nms.h"

using namespace at;

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class ROIAlignFunction : public torch::autograd::Function<ROIAlignFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input,
      Variable rois,
      const double spatial_scale,
      const int64_t pooled_height,
      const int64_t pooled_width,
      const int64_t sampling_ratio) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["sampling_ratio"] = sampling_ratio;
    ctx->saved_data["batch_size"] = input.size(0);
    ctx->saved_data["channels"] = input.size(1);
    ctx->saved_data["height"] = input.size(2);
    ctx->saved_data["width"] = input.size(3);
    ctx->save_for_backward({rois});
    auto result = ROIAlign_forward(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio);
    return {result};
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto grad_in = ROIAlign_backward(
        grad_output[0],
        rois,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toInt(),
        ctx->saved_data["pooled_width"].toInt(),
        ctx->saved_data["batch_size"].toInt(),
        ctx->saved_data["channels"].toInt(),
        ctx->saved_data["height"].toInt(),
        ctx->saved_data["width"].toInt(),
        ctx->saved_data["sampling_ratio"].toInt());
    return {
        grad_in, Variable(), Variable(), Variable(), Variable(), Variable()};
  }
};

Tensor roi_align(
    const Tensor& input,
    const Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio) {
  return ROIAlignFunction::apply(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio)[0];
}

class ROIPoolFunction : public torch::autograd::Function<ROIPoolFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input,
      Variable rois,
      const double spatial_scale,
      const int64_t pooled_height,
      const int64_t pooled_width) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["batch_size"] = input.size(0);
    ctx->saved_data["channels"] = input.size(1);
    ctx->saved_data["height"] = input.size(2);
    ctx->saved_data["width"] = input.size(3);
    auto result = ROIPool_forward(
        input, rois, spatial_scale, pooled_height, pooled_width);
    auto output = std::get<0>(result);
    auto argmax = std::get<1>(result);
    ctx->save_for_backward({rois, argmax});
    ctx->mark_non_differentiable({argmax});
    return {output, argmax};
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];
    auto argmax = saved[1];
    auto grad_in = ROIPool_backward(
        grad_output[0],
        rois,
        argmax,
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["pooled_height"].toInt(),
        ctx->saved_data["pooled_width"].toInt(),
        ctx->saved_data["batch_size"].toInt(),
        ctx->saved_data["channels"].toInt(),
        ctx->saved_data["height"].toInt(),
        ctx->saved_data["width"].toInt());
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

std::tuple<at::Tensor, at::Tensor> roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width) {
  auto result = ROIPoolFunction::apply(
      input, rois, spatial_scale, pooled_height, pooled_width);
  return {result[0], result[1]};
}

static auto registry =
    torch::jit::RegisterOperators()
        .op("torchvision::nms", &nms)
        .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor",
            &roi_align)
        .op("torchvision::roi_pool", &roi_pool);
