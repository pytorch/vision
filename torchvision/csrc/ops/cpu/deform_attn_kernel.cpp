#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {
at::Tensor ms_deform_attn_forward_kernel(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const int64_t im2col_step) {
  TORCH_CHECK(false, "Deformable attention is only supported on CUDA for now.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ms_deform_attn_backward_kernel(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    const int64_t im2col_step) {
  TORCH_CHECK(false, "Deformable attention is only supported on CUDA for now.");
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_attn"),
      TORCH_FN(ms_deform_attn_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_deform_attn_backward"),
      TORCH_FN(ms_deform_attn_backward_kernel));
}
} // namespace ops
} // namespace vision
