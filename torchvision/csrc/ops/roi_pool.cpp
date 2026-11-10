#include "roi_pool.h"

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/headeronly/version.h>

#include <array>

namespace vision {
namespace ops {

using torch::stable::Tensor;

std::tuple<Tensor, Tensor> roi_pool(
    const Tensor& input,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  std::array<StableIValue, 5> stack{
      torch::stable::detail::from(input),
      torch::stable::detail::from(rois),
      torch::stable::detail::from(spatial_scale),
      torch::stable::detail::from(pooled_height),
      torch::stable::detail::from(pooled_width)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::roi_pool", "", stack.data(), TORCH_ABI_VERSION));
  return std::make_tuple(
      torch::stable::detail::to<Tensor>(stack[0]),
      torch::stable::detail::to<Tensor>(stack[1]));
}

namespace detail {

Tensor _roi_pool_backward(
    const Tensor& grad,
    const Tensor& rois,
    const Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  std::array<StableIValue, 10> stack{
      torch::stable::detail::from(grad),
      torch::stable::detail::from(rois),
      torch::stable::detail::from(argmax),
      torch::stable::detail::from(spatial_scale),
      torch::stable::detail::from(pooled_height),
      torch::stable::detail::from(pooled_width),
      torch::stable::detail::from(batch_size),
      torch::stable::detail::from(channels),
      torch::stable::detail::from(height),
      torch::stable::detail::from(width)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "torchvision::_roi_pool_backward", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

} // namespace detail

STABLE_TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "roi_pool(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width) -> (Tensor, Tensor)");
  m.def(
      "_roi_pool_backward(Tensor grad, Tensor rois, Tensor argmax, float spatial_scale, SymInt pooled_height, SymInt pooled_width, SymInt batch_size, SymInt channels, SymInt height, SymInt width) -> Tensor");
}

} // namespace ops
} // namespace vision
