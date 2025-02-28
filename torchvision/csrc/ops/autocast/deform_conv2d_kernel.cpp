#include "../deform_conv2d.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

template <c10::DispatchKey autocast_key, c10::DeviceType device_type>
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
  c10::impl::ExcludeDispatchKeyGuard no_autocast(autocast_key);
  return deform_conv2d(
             at::autocast::cached_cast(at::kFloat, input, device_type),
             at::autocast::cached_cast(at::kFloat, weight, device_type),
             at::autocast::cached_cast(at::kFloat, offset, device_type),
             at::autocast::cached_cast(at::kFloat, mask, device_type),
             at::autocast::cached_cast(at::kFloat, bias, device_type),
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

TORCH_LIBRARY_IMPL(torchvision, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
      TORCH_FN((deform_conv2d_autocast<
                c10::DispatchKey::Autocast,
                c10::DeviceType::CUDA>)));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
      TORCH_FN((deform_conv2d_autocast<
                c10::DispatchKey::AutocastCPU,
                c10::DeviceType::CPU>)));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastXPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
      TORCH_FN((deform_conv2d_autocast<
                c10::DispatchKey::AutocastXPU,
                c10::DeviceType::XPU>)));
}

} // namespace ops
} // namespace vision
