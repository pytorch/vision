#include "decode_heic.h"

#if HEIC_FOUND
#include "libheif/heif.h"
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
  heif_context* ctx = heif_context_alloc();
  heif_context_read_from_memory(
      ctx, encoded_data.data_ptr<uint8_t>(), encoded_data.numel(), nullptr);

  // get a handle to the primary image
  heif_image_handle* handle;
  heif_context_get_primary_image_handle(ctx, &handle);
  int width = heif_image_handle_get_width(handle);
  int height = heif_image_handle_get_height(handle);

  // decode the image and convert colorspace to RGB, saved as 24bit interleaved
  heif_image* img;
  heif_decode_image(
      handle, &img, heif_colorspace_RGB, heif_chroma_interleaved_RGB, nullptr);

  int stride;
  uint8_t* data = (uint8_t*)heif_image_get_plane_readonly(
      img, heif_channel_interleaved, &stride);
  auto out = torch::from_blob(data, {height, width, 3}, torch::kUInt8);
  return out.permute({2, 0, 1});

  // TODO
  // clean up resources
  // heif_image_release(img);
  // heif_image_handle_release(handle);
  // heif_context_free(ctx);
}
#endif // HEIC_FOUND

} // namespace image
} // namespace vision
