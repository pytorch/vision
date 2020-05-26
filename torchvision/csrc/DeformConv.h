#pragma once

#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif
#ifdef WITH_HIP
#include "hip/vision_cuda.h"
#endif

at::Tensor DeformConv2d_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& bias,
    const std::pair<int, int>& stride,
    const std::pair<int, int>& padding,
    const std::pair<int, int>& dilation,
    const int groups,
    const int offset_groups) {
  if (input.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return DeformConv2d_forward_cuda(
        input.contiguous(),
        weight.contiguous(),
        offset.contiguous(),
        bias.contiguous(),
        stride,
        padding,
        dilation,
        groups,
        offset_groups);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return DeformConv2d_forward_cpu(
      input.contiguous(),
      weight.contiguous(),
      offset.contiguous(),
      bias.contiguous(),
      stride,
      padding,
      dilation,
      groups,
      offset_groups);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> DeformConv2d_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& bias,
    const std::pair<int, int>& stride,
    const std::pair<int, int>& padding,
    const std::pair<int, int>& dilation,
    const int groups,
    const int offset_groups) {
  if (grad.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return DeformConv2d_backward_cuda(
        grad.contiguous(),
        input.contiguous(),
        weight.contiguous(),
        offset.contiguous(),
        bias.contiguous(),
        stride,
        padding,
        dilation,
        groups,
        offset_groups);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return DeformConv2d_backward_cpu(
      grad.contiguous(),
      input.contiguous(),
      weight.contiguous(),
      offset.contiguous(),
      bias.contiguous(),
      stride,
      padding,
      dilation,
      groups,
      offset_groups);
}

class DeformConv2dFunction
    : public torch::autograd::Function<DeformConv2dFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::Variable input,
      torch::autograd::Variable weight,
      torch::autograd::Variable offset,
      torch::autograd::Variable bias,
      int64_t stride_h,
      int64_t stride_w,
      int64_t pad_h,
      int64_t pad_w,
      int64_t dilation_h,
      int64_t dilation_w,
      int64_t groups,
      int64_t offset_groups) {
    auto output = DeformConv2d_forward(
        input,
        weight,
        offset,
        bias,
        {stride_h, stride_w},
        {pad_h, pad_w},
        {dilation_h, dilation_w},
        groups,
        offset_groups);

    ctx->save_for_backward({input, weight, offset, bias});
    ctx->saved_data["stride_h"] = stride_h;
    ctx->saved_data["stride_w"] = stride_w;
    ctx->saved_data["pad_h"] = pad_h;
    ctx->saved_data["pad_w"] = pad_w;
    ctx->saved_data["dilation_h"] = dilation_h;
    ctx->saved_data["dilation_w"] = dilation_w;
    ctx->saved_data["groups"] = groups;
    ctx->saved_data["offset_groups"] = offset_groups;

    return {
        output,
    };
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto offset = saved[2];
    auto bias = saved[3];

    auto stride_h = ctx->saved_data["stride_h"].toInt();
    auto stride_w = ctx->saved_data["stride_w"].toInt();
    auto pad_h = ctx->saved_data["pad_h"].toInt();
    auto pad_w = ctx->saved_data["pad_w"].toInt();
    auto dilation_h = ctx->saved_data["dilation_h"].toInt();
    auto dilation_w = ctx->saved_data["dilation_w"].toInt();
    auto groups = ctx->saved_data["groups"].toInt();
    auto offset_groups = ctx->saved_data["offset_groups"].toInt();

    auto grads = DeformConv2d_backward(
        grad_output[0],
        input,
        weight,
        offset,
        bias,
        {stride_h, stride_w},
        {pad_h, pad_w},
        {dilation_h, dilation_w},
        groups,
        offset_groups);
    auto grad_input = std::get<0>(grads);
    auto grad_weight = std::get<1>(grads);
    auto grad_offset = std::get<2>(grads);
    auto grad_bias = std::get<3>(grads);

    return {
        grad_input,
        grad_weight,
        grad_offset,
        grad_bias,
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
    };
  }
};

at::Tensor deform_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups) {
  auto result = DeformConv2dFunction::apply(
      input,
      weight,
      offset,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      offset_groups);
  return result[0];
}
