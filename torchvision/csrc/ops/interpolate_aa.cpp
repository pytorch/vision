#include "interpolate_aa.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor interpolate_linear_aa(
    const at::Tensor& input, // Input image
    at::IntArrayRef output_size, // Output image size
    bool align_corners) // The flag to align corners
{
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_interpolate_linear_aa", "")
          .typed<decltype(interpolate_linear_aa)>();
  return op.call(input, output_size, align_corners);
}

at::Tensor interpolate_bicubic_aa(
    const at::Tensor& input, // Input image
    at::IntArrayRef output_size, // Output image size
    bool align_corners) // The flag to align corners
{
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_interpolate_bicubic_aa", "")
          .typed<decltype(_interpolate_bicubic_aa)>();
  return op.call(input, output_size, align_corners);
}

namespace detail {

// TODO: Implement backward function
// at::Tensor _interpolate_linear_aa_backward(
//     const at::Tensor& grad,
//     at::IntArrayRef output_size,
//     bool align_corners)
// {
//   return at::Tensor();
// }

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_interpolate_linear_aa(Tensor input, int[] output_size, bool align_corners) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_interpolate_bicubic_aa(Tensor input, int[] output_size, bool align_corners) -> Tensor"));
  // TODO: Implement backward function
  // m.def(TORCH_SELECTIVE_SCHEMA(
  //     "torchvision::_interpolate_linear_aa_backward(Tensor grad, Tensor rois,
  //     float spatial_scale, int pooled_height, int pooled_width, int
  //     batch_size, int channels, int height, int width, int sampling_ratio,
  //     bool aligned) -> Tensor"));
}

} // namespace ops
} // namespace vision
