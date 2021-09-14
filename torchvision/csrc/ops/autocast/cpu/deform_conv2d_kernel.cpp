#include "../../deform_conv2d.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

at::Tensor deform_conv2d_autocast(
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
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastCPU);
  return deform_conv2d(
             at::autocast::cached_cast(at::kFloat, input, c10::DeviceType::CPU),
             at::autocast::cached_cast(
                 at::kFloat, weight, c10::DeviceType::CPU),
             at::autocast::cached_cast(
                 at::kFloat, offset, c10::DeviceType::CPU),
             at::autocast::cached_cast(at::kFloat, mask, c10::DeviceType::CPU),
             at::autocast::cached_cast(at::kFloat, bias, c10::DeviceType::CPU),
             stride_h,
             stride_w,
             pad_h,
             pad_w,
             dilation_h,
             dilation_w,
             groups,
             offset_groups,
             use_mask)
      .to(input.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, AutocastCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
      TORCH_FN(deform_conv2d_autocast));
}

} // namespace ops
} // namespace vision
