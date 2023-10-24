#include "../deform_conv2d.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

class DeformConv2dFunction
    : public torch::autograd::Function<DeformConv2dFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& weight,
      const torch::autograd::Variable& offset,
      const torch::autograd::Variable& mask,
      const torch::autograd::Variable& bias,
      c10::SymInt stride_h,
      c10::SymInt stride_w,
      c10::SymInt pad_h,
      c10::SymInt pad_w,
      c10::SymInt dilation_h,
      c10::SymInt dilation_w,
      c10::SymInt groups,
      c10::SymInt offset_groups,
      bool use_mask) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto output = deform_conv2d_symint(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        use_mask);

    ctx->save_for_backward({input, weight, offset, mask, bias});
    ctx->saved_data["stride_h"] = stride_h;
    ctx->saved_data["stride_w"] = stride_w;
    ctx->saved_data["pad_h"] = pad_h;
    ctx->saved_data["pad_w"] = pad_w;
    ctx->saved_data["dilation_h"] = dilation_h;
    ctx->saved_data["dilation_w"] = dilation_w;
    ctx->saved_data["groups"] = groups;
    ctx->saved_data["offset_groups"] = offset_groups;
    ctx->saved_data["use_mask"] = use_mask;

    return {
        output,
    };
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto offset = saved[2];
    auto mask = saved[3];
    auto bias = saved[4];

    auto stride_h = ctx->saved_data["stride_h"].toSymInt();
    auto stride_w = ctx->saved_data["stride_w"].toSymInt();
    auto pad_h = ctx->saved_data["pad_h"].toSymInt();
    auto pad_w = ctx->saved_data["pad_w"].toSymInt();
    auto dilation_h = ctx->saved_data["dilation_h"].toSymInt();
    auto dilation_w = ctx->saved_data["dilation_w"].toSymInt();
    auto groups = ctx->saved_data["groups"].toSymInt();
    auto offset_groups = ctx->saved_data["offset_groups"].toSymInt();
    auto use_mask = ctx->saved_data["use_mask"].toBool();

    auto grads = detail::_deform_conv2d_backward_symint(
        grad_output[0],
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        use_mask);
    auto grad_input = std::get<0>(grads);
    auto grad_weight = std::get<1>(grads);
    auto grad_offset = std::get<2>(grads);
    auto grad_mask = std::get<3>(grads);
    auto grad_bias = std::get<4>(grads);

    return {
        grad_input,
        grad_weight,
        grad_offset,
        grad_mask,
        grad_bias,
        torch::autograd::Variable(),
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

// TODO: There should be an easier way to do this
class DeformConv2dBackwardFunction
    : public torch::autograd::Function<DeformConv2dBackwardFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& grad,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& weight,
      const torch::autograd::Variable& offset,
      const torch::autograd::Variable& mask,
      const torch::autograd::Variable& bias,
      c10::SymInt stride_h,
      c10::SymInt stride_w,
      c10::SymInt pad_h,
      c10::SymInt pad_w,
      c10::SymInt dilation_h,
      c10::SymInt dilation_w,
      c10::SymInt groups,
      c10::SymInt offset_groups,
      bool use_mask) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto result = detail::_deform_conv2d_backward_symint(
        grad,
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        use_mask);

    auto grad_input = std::get<0>(result);
    auto grad_weight = std::get<1>(result);
    auto grad_offset = std::get<2>(result);
    auto grad_mask = std::get<3>(result);
    auto grad_bias = std::get<4>(result);

    return {
        grad_input,
        grad_weight,
        grad_offset,
        grad_mask,
        grad_bias,
    };
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    TORCH_CHECK(0, "double backwards on deform_conv2d not supported");
  }
};

at::Tensor deform_conv2d_autograd(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    c10::SymInt stride_h,
    c10::SymInt stride_w,
    c10::SymInt pad_h,
    c10::SymInt pad_w,
    c10::SymInt dilation_h,
    c10::SymInt dilation_w,
    c10::SymInt groups,
    c10::SymInt offset_groups,
    bool use_mask) {
  return DeformConv2dFunction::apply(
      input,
      weight,
      offset,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      offset_groups,
      use_mask)[0];
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv2d_backward_autograd(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    c10::SymInt stride_h,
    c10::SymInt stride_w,
    c10::SymInt pad_h,
    c10::SymInt pad_w,
    c10::SymInt dilation_h,
    c10::SymInt dilation_w,
    c10::SymInt groups,
    c10::SymInt offset_groups,
    bool use_mask) {
  auto result = DeformConv2dBackwardFunction::apply(
      grad,
      input,
      weight,
      offset,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      offset_groups,
      use_mask);

  return std::make_tuple(result[0], result[1], result[2], result[3], result[4]);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
      TORCH_FN(deform_conv2d_autograd));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_deform_conv2d_backward"),
      TORCH_FN(deform_conv2d_backward_autograd));
}

} // namespace ops
} // namespace vision
