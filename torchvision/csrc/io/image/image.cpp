#include "image.h"

#include "../../StableABICompat.h"

namespace vision {
namespace image {

// Register operators using stable ABI macros
STABLE_TORCH_LIBRARY(image, m) {
  m.def("decode_gif(Tensor data) -> Tensor");
  m.def(
      "decode_png(Tensor data, int mode, bool apply_exif_orientation=False) -> Tensor");
  m.def("encode_png(Tensor data, int compression_level) -> Tensor");
  m.def(
      "decode_jpeg(Tensor data, int mode, bool apply_exif_orientation=False) -> Tensor");
  m.def("decode_webp(Tensor encoded_data, int mode) -> Tensor");
  m.def("encode_jpeg(Tensor data, int quality) -> Tensor");
  m.def("read_file(str filename) -> Tensor");
  m.def("write_file(str filename, Tensor data) -> ()");
  m.def(
      "decode_image(Tensor data, int mode, bool apply_exif_orientation=False) -> Tensor");
  // CUDA JPEG ops are disabled when building with stable ABI
  // (TORCH_TARGET_VERSION) because the stable ABI doesn't expose raw CUDA
  // streams needed by nvJPEG. m.def("decode_jpegs_cuda(Tensor[] encoded_jpegs,
  // int mode, Device device) -> Tensor[]"); m.def("encode_jpegs_cuda(Tensor[]
  // decoded_jpegs, int quality) -> Tensor[]");
  m.def("_jpeg_version() -> int");
  m.def("_is_compiled_against_turbo() -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(image, CPU, m) {
  m.impl("decode_gif", TORCH_BOX(&decode_gif));
  m.impl("decode_png", TORCH_BOX(&decode_png));
  m.impl("encode_png", TORCH_BOX(&encode_png));
  m.impl("decode_jpeg", TORCH_BOX(&decode_jpeg));
  m.impl("decode_webp", TORCH_BOX(&decode_webp));
  m.impl("encode_jpeg", TORCH_BOX(&encode_jpeg));
  m.impl("decode_image", TORCH_BOX(&decode_image));
}

// Ops without tensor inputs need BackendSelect dispatch
STABLE_TORCH_LIBRARY_IMPL(image, BackendSelect, m) {
  m.impl("read_file", TORCH_BOX(&read_file));
  m.impl("write_file", TORCH_BOX(&write_file));
  m.impl("_jpeg_version", TORCH_BOX(&_jpeg_version));
  m.impl("_is_compiled_against_turbo", TORCH_BOX(&_is_compiled_against_turbo));
}

// CUDA JPEG is disabled when building with stable ABI (TORCH_TARGET_VERSION)
// because the stable ABI doesn't expose raw CUDA streams needed by nvJPEG.
// STABLE_TORCH_LIBRARY_IMPL(image, CUDA, m) {
//   m.impl("decode_jpegs_cuda", TORCH_BOX(&decode_jpegs_cuda));
//   m.impl("encode_jpegs_cuda", TORCH_BOX(&encode_jpegs_cuda));
// }

} // namespace image
} // namespace vision
