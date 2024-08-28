#include "decode_heic.h"

#if HEIC_FOUND
// #include "libheif/heif.h"
#include "libheif/heif_cxx.h"
#endif // HEIC_FOUND

namespace vision {
namespace image {

#if !HEIC_FOUND
torch::Tensor decode_heic(const torch::Tensor& encoded_data, ImageReadMode mode) {
  TORCH_CHECK(
      false, "decode_heic: torchvision not compiled with libheif support");
}
#else

torch::Tensor decode_heic(const torch::Tensor& encoded_data, ImageReadMode mode) {
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

    if (mode != IMAGE_READ_MODE_UNCHANGED && mode != IMAGE_READ_MODE_RGB &&
      mode != IMAGE_READ_MODE_RGB_ALPHA) {
    // Other modes aren't supported, but we don't error or even warn because we
    // have generic entry points like decode_image which may support all modes,
    // it just depends on the underlying decoder.
    mode = IMAGE_READ_MODE_UNCHANGED;
  }

  // If return_rgb is false it means we return rgba - nothing else.
  auto return_rgb = true;

  int height = 0;
  int width = 0;
  int num_channels = 0;
  int stride = 0;
  uint8_t* decoded_data = nullptr;
  heif::Image img;

  try {
    heif::Context ctx;
    ctx.read_from_memory_without_copy(
        encoded_data.data_ptr<uint8_t>(), encoded_data.numel());

    heif::ImageHandle handle = ctx.get_primary_image_handle();
    return_rgb = (mode == IMAGE_READ_MODE_RGB ||
       (mode == IMAGE_READ_MODE_UNCHANGED && !handle.has_alpha_channel()));

    height = handle.get_height();
    width = handle.get_width();

    num_channels = return_rgb ? 3 : 4;
    auto chroma = return_rgb? heif_chroma_interleaved_RGB : heif_chroma_interleaved_RGBA;
    img = handle.decode_image(heif_colorspace_RGB, chroma);

    decoded_data = img.get_plane(heif_channel_interleaved, &stride);
  } catch (const heif::Error& err) {
    TORCH_CHECK(false, "decode_heif failed: ", err.get_message());
  }
  TORCH_CHECK(decoded_data != nullptr, "Something went wrong during decoding.");

  auto out = torch::empty({height, width, num_channels}, torch::kUInt8);
  auto out_ptr = out.data_ptr<uint8_t>();

  // decoded_data is *almost* the raw decoded data, but not quite: for some
  // images, there may be some padding at the end of each row, i.e. when stride
  // != row_size. So we can't copy decoded_data into the tensor's memory
  // directly, we have to copy row by row. Oh, and if you think you can take a
  // shortcut when stride == row_size and just do:
  // out =  torch::from_blob(decoded_data, ...)
  // you can't, because decoded_data is owned by the heif::Image object and gets
  // freed when it gets out of scope!
  auto row_size = width * num_channels;
  for (auto i = 0; i < height; i++) {
    memcpy(out_ptr, decoded_data, row_size);
    out_ptr += row_size;
    decoded_data += stride;
  }
  return out.permute({2, 0, 1});
}
#endif // HEIC_FOUND

} // namespace image
} // namespace vision
