#include "deform_conv2d.h"

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/headeronly/version.h>

#include <array>

namespace vision {
namespace ops {

using torch::stable::Tensor;

Tensor deform_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& mask,
    const Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  std::array<StableIValue, 14> stack{
      torch::stable::detail::from(input),
      torch::stable::detail::from(weight),
      torch::stable::detail::from(offset),
      torch::stable::detail::from(mask),
      torch::stable::detail::from(bias),
      torch::stable::detail::from(stride_h),
      torch::stable::detail::from(stride_w),
      torch::stable::detail::from(pad_h),
      torch::stable::detail::from(pad_w),
      torch::stable::detail::from(dilation_h),
      torch::stable::detail::from(dilation_w),
      torch::stable::detail::from(groups),
      torch::stable::detail::from(offset_groups),
      torch::stable::detail::from(use_mask)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::deform_conv2d", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

namespace detail {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _deform_conv2d_backward(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& mask,
    const Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  std::array<StableIValue, 15> stack{
      torch::stable::detail::from(grad),
      torch::stable::detail::from(input),
      torch::stable::detail::from(weight),
      torch::stable::detail::from(offset),
      torch::stable::detail::from(mask),
      torch::stable::detail::from(bias),
      torch::stable::detail::from(stride_h),
      torch::stable::detail::from(stride_w),
      torch::stable::detail::from(pad_h),
      torch::stable::detail::from(pad_w),
      torch::stable::detail::from(dilation_h),
      torch::stable::detail::from(dilation_w),
      torch::stable::detail::from(groups),
      torch::stable::detail::from(offset_groups),
      torch::stable::detail::from(use_mask)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::_deform_conv2d_backward",
      "",
      stack.data(),
      TORCH_ABI_VERSION));
  return std::make_tuple(
      torch::stable::detail::to<Tensor>(stack[0]),
      torch::stable::detail::to<Tensor>(stack[1]),
      torch::stable::detail::to<Tensor>(stack[2]),
      torch::stable::detail::to<Tensor>(stack[3]),
      torch::stable::detail::to<Tensor>(stack[4]));
}

} // namespace detail

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, SymInt stride_h, SymInt stride_w, SymInt pad_h, SymInt pad_w, SymInt dilation_h, SymInt dilation_w, SymInt groups, SymInt offset_groups, bool use_mask) -> Tensor");
  m.def(
      "_deform_conv2d_backward(Tensor grad, Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, SymInt stride_h, SymInt stride_w, SymInt pad_h, SymInt pad_w, SymInt dilation_h, SymInt dilation_w, SymInt groups, SymInt offset_groups, bool use_mask) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

} // namespace ops
} // namespace vision
