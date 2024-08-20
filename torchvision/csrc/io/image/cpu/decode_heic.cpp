#include "decode_heic.h"

#if HEIC_FOUND
// #include "libheif/heif.h"
#include "libheif/heif_cxx.h"
#endif // HEIC_FOUND

namespace vision {
namespace image {

#if !HEIC_FOUND
torch::Tensor decode_heic(const torch::Tensor& data) {
  TORCH_CHECK(
      false, "decode_heic: torchvision not compiled with libheif support");
}
#else

torch::Tensor decode_heic(const torch::Tensor& encoded_data) {
  TORCH_CHECK(encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  TORCH_CHECK(
      encoded_data.dtype() == torch::kU8,
      "Input tensor must have uint8 data type, got ",
      encoded_data.dtype());
  TORCH_CHECK(
      encoded_data.dim() == 1,
      "Input tensor must be 1-dimensional, got ",
      encoded_data.dim(),
      " dims.");

  heif::Context ctx;
  ctx.read_from_memory_without_copy(
      encoded_data.data_ptr<uint8_t>(), encoded_data.numel());

  heif::ImageHandle handle = ctx.get_primary_image_handle();
  heif::Image img =
      handle.decode_image(heif_colorspace_RGB, heif_chroma_interleaved_RGB);

  int stride;
  uint8_t* decoded_data = img.get_plane(heif_channel_interleaved, &stride);
  TORCH_CHECK(decoded_data != nullptr, "Something went wrong during decoding.");

  auto out =
      torch::empty({handle.get_height(), handle.get_width(), 3}, torch::kUInt8);
  auto out_ptr = out.data_ptr<uint8_t>();

  // decoded_data is *almost* the raw decoded data: for some image, there may be
  // some padding at the end of each row, i.e. when stride != row_size. So we
  // can't copy decoded_data into the tensor's memory directly, we have to copy
  // row by row.
  // Oh, and if you think you can take a shortcut when stride == row_size and
  // just do:
  // out = torch::from_blob(decoded_data, ...)
  // you can't, because decoded_data is owned by the heif::Image object and gets
  // freed when it gets out of scope!
  auto row_size = handle.get_width() * 3;
  for (auto i = 0; i < handle.get_height(); i++) {
    memcpy(out_ptr, decoded_data, row_size);
    out_ptr += row_size;
    decoded_data += stride;
  }
  return out.permute({2, 0, 1});
}
#endif // HEIC_FOUND

} // namespace image
} // namespace vision
