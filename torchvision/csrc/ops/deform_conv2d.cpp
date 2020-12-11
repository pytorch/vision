#include "deform_conv2d.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor deform_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::deform_conv2d", "")
                       .typed<decltype(deform_conv2d)>();
  return op.call(
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
}

namespace detail {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_deform_conv2d_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_deform_conv2d_backward", "")
          .typed<decltype(_deform_conv2d_backward)>();
  return op.call(
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
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> Tensor");
  m.def(
      "_deform_conv2d_backward(Tensor grad, Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

} // namespace ops
} // namespace vision
