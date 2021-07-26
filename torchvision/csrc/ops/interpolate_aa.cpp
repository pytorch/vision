#include "interpolate_aa.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor _interpolate_bilinear2d_aa(
    const at::Tensor& input, // Input image
    at::IntArrayRef output_size, // Output image size
    bool align_corners) // The flag to align corners
{
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_interpolate_bilinear2d_aa", "")
          .typed<decltype(_interpolate_bilinear2d_aa)>();
  return op.call(input, output_size, align_corners);
}

at::Tensor _interpolate_bicubic_aa(
    const at::Tensor& input, // Input image
    at::IntArrayRef output_size, // Output image size
    bool align_corners) // The flag to align corners
{
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_interpolate_bicubic2d_aa", "")
          .typed<decltype(_interpolate_bicubic2d_aa)>();
  return op.call(input, output_size, align_corners);
}

namespace detail {

at::Tensor _interpolate_bilinear2d_aa_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchvision::_interpolate_bilinear2d_aa_backward", "")
          .typed<decltype(_interpolate_bilinear2d_aa_backward)>();
  return op.call(grad_output, output_size, output_size, align_corners);
}

at::Tensor _interpolate_bicubic2d_aa_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchvision::_interpolate_bicubic2d_aa_backward", "")
          .typed<decltype(_interpolate_bicubic2d_aa_backward)>();
  return op.call(grad_output, output_size, output_size, align_corners);
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_interpolate_bilinear2d_aa(Tensor input, int[] output_size, bool align_corners) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_interpolate_bicubic2d_aa(Tensor input, int[] output_size, bool align_corners) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_interpolate_bilinear2d_aa_backward(Tensor input, int[] output_size, int[] input_size, bool align_corners) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_interpolate_bicubic2d_aa_backward(Tensor input, int[] output_size, int[] input_size, bool align_corners) -> Tensor"));
}

} // namespace ops
} // namespace vision
