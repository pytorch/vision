#include "decode_heic.h"
#include "../common.h"

#if HEIC_FOUND
#include "libheif/heif_cxx.h"
#endif // HEIC_FOUND

namespace vision {
namespace image {

#if !HEIC_FOUND
torch::Tensor decode_heic(
    const torch::Tensor& encoded_data,
    ImageReadMode mode) {
  TORCH_CHECK(
      false, "decode_heic: torchvision not compiled with libheif support");
}
#else

torch::Tensor decode_heic(
    const torch::Tensor& encoded_data,
    ImageReadMode mode) {
  validate_encoded_data(encoded_data);

  auto return_rgb = true;

  int height = 0;
  int width = 0;
  int num_channels = 0;
  int stride = 0;
  uint8_t* decoded_data = nullptr;
  heif::Image img;
  int bit_depth = 0;

  try {
    heif::Context ctx;
    ctx.read_from_memory_without_copy(
        encoded_data.data_ptr<uint8_t>(), encoded_data.numel());

    // TODO properly error on (or support) image sequences. Right now, I think
    // this function will always return the first image in a sequence, which is
    // inconsistent with decode_gif (which returns a batch) and with decode_avif
    // (which errors loudly).
    // Why? I'm struggling to make sense of
    // ctx.get_number_of_top_level_images(). It disagrees with libavif's
    // imageCount. For example on some of the libavif test images:
    //
    // - colors-animated-12bpc-keyframes-0-2-3.avif
    //   avif num images = 5
    //   heif num images = 1  // Why is this 1 when clearly this is supposed to
    //                           be a sequence?
    // - sofa_grid1x5_420.avif
    //   avif num images = 1
    //   heif num images = 6  // If we were to error here we won't be able to
    //                           decode this image which is otherwise properly
    //                           decoded by libavif.
    // I can't find a libheif function that does what we need here, or at least
    // that agrees with libavif.

    // TORCH_CHECK(
    //     ctx.get_number_of_top_level_images() == 1,
    //     "heic file contains more than one image");

    heif::ImageHandle handle = ctx.get_primary_image_handle();
    bit_depth = handle.get_luma_bits_per_pixel();

    return_rgb =
        should_this_return_rgb_or_rgba_let_me_know_in_the_comments_down_below_guys_see_you_in_the_next_video(
            mode, handle.has_alpha_channel());

    height = handle.get_height();
    width = handle.get_width();

    num_channels = return_rgb ? 3 : 4;
    heif_chroma chroma;
    if (bit_depth == 8) {
      chroma = return_rgb ? heif_chroma_interleaved_RGB
                          : heif_chroma_interleaved_RGBA;
    } else {
      // TODO: This, along with our 10bits -> 16bits range mapping down below,
      // may not work on BE platforms
      chroma = return_rgb ? heif_chroma_interleaved_RRGGBB_LE
                          : heif_chroma_interleaved_RRGGBBAA_LE;
    }

    img = handle.decode_image(heif_colorspace_RGB, chroma);

    decoded_data = img.get_plane(heif_channel_interleaved, &stride);
  } catch (const heif::Error& err) {
    // We need this try/catch block and call TORCH_CHECK, because libheif may
    // otherwise throw heif::Error that would just be reported as "An unknown
    // exception occurred" when we move back to Python.
    TORCH_CHECK(false, "decode_heif failed: ", err.get_message());
  }
  TORCH_CHECK(decoded_data != nullptr, "Something went wrong during decoding.");

  auto dtype = (bit_depth == 8) ? torch::kUInt8 : at::kUInt16;
  auto out = torch::empty({height, width, num_channels}, dtype);
  uint8_t* out_ptr = (uint8_t*)out.data_ptr();

  // decoded_data is *almost* the raw decoded data, but not quite: for some
  // images, there may be some padding at the end of each row, i.e. when stride
  // != row_size_in_bytes. So we can't copy decoded_data into the tensor's
  // memory directly, we have to copy row by row. Oh, and if you think you can
  // take a shortcut when stride == row_size_in_bytes and just do:
  // out =  torch::from_blob(decoded_data, ...)
  // you can't, because decoded_data is owned by the heif::Image object and it
  // gets freed when it gets out of scope!
  auto row_size_in_bytes = width * num_channels * ((bit_depth == 8) ? 1 : 2);
  for (auto h = 0; h < height; h++) {
    memcpy(
        out_ptr + h * row_size_in_bytes,
        decoded_data + h * stride,
        row_size_in_bytes);
  }
  if (bit_depth > 8) {
    // Say bit depth is 10. decodec_data and out_ptr contain 10bits values
    // over 2 bytes, stored into uint16_t. In torchvision a uint16 value is
    // expected to be in [0, 2**16), so we have to map the 10bits value to that
    // range. Note that other libraries like libavif do that mapping
    // automatically.
    // TODO: It's possible to avoid the memcpy call above in this case, and do
    // the copy at the same time as the conversation. Whether it's worth it
    // should be benchmarked.
    auto out_ptr_16 = (uint16_t*)out_ptr;
    for (auto p = 0; p < height * width * num_channels; p++) {
      out_ptr_16[p] <<= (16 - bit_depth);
    }
  }
  return out.permute({2, 0, 1});
}
#endif // HEIC_FOUND

} // namespace image
} // namespace vision
