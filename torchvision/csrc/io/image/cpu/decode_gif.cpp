#include "decode_gif.h"
#include "giflib/gif_lib.h"

namespace vision {
namespace image {

typedef struct reader_helper_t {
  uint8_t const* encoded_data; // input tensor data pointer
  size_t encoded_data_size; // size of input tensor in bytes
  int num_bytes_read; // number of bytes read so far in the tensor
} reader_helper_t;

// That function is used by GIFLIB routines to read the encoded bytes.
// This reads `len` bytes and writes them into `buf`. The data is read from the
// input tensor passed to decode_gif() starting at the `num_bytes_read`
// position.
int read_from_tensor(GifFileType* gifFile, GifByteType* buf, int len) {
  // the UserData field was set in DGifOpen()
  reader_helper_t* reader_helper = (reader_helper_t*)gifFile->UserData;

  auto i = 0;
  auto num_bytes_to_read = std::min(
      len,
      (int)(reader_helper->encoded_data_size - reader_helper->num_bytes_read));
  while (i < num_bytes_to_read) {
    buf[i] = reader_helper->encoded_data[reader_helper->num_bytes_read + i];
    i++;
  }
  reader_helper->num_bytes_read += i;
  return i;
}

torch::Tensor decode_gif(const torch::Tensor& encoded_data) {
  // LibGif docs: https://giflib.sourceforge.net/intro.html
  // Refer over there for more details on the libgif API, API ref, and a
  // detailed description of the GIF format.

  int error = D_GIF_SUCCEEDED;

  // We're using DGidOpen. The other entrypoints of libgif are
  // DGifOpenFileName and DGifOpenFileHandle but we don't want to use those,
  // since we need to read the encoded bytes from a tensor of encoded bytes, not
  // from a file (for consistency with existing jpeg and png decoders). Using
  // DGifOpen is the only way to read from a custom source.
  // For that we need to provide a reader function `read_from_tensor` that
  // reads from the tensor, and we have to keep track of the number of bytes
  // read so far: this is why we need the reader_helper struct.

  // TODO: We are potentially doing an unnecessary copy of the encoded bytes:
  // - 1 copy in from file to tensor (in read_file())
  // - 1 copy from tensor to GIFLIB buffers (in read_from_tensor())
  // Since we're vendoring GIFLIB we can potentially modify the calls to
  // InternalRead() and just set the `buf` pointer to the tensor data directly.
  // That might even save allocation of those buffers.
  // If we do that, we'd have to make sure the buffers are never written to by
  // GIFLIB, otherwise we'd be overridding the tensor data.
  reader_helper_t reader_helper;
  reader_helper.encoded_data = encoded_data.data_ptr<uint8_t>();
  reader_helper.encoded_data_size = encoded_data.numel();
  reader_helper.num_bytes_read = 0;
  GifFileType* gifFile =
      DGifOpen((void*)&reader_helper, read_from_tensor, &error);

  TORCH_CHECK(
      (gifFile != nullptr) && (error == D_GIF_SUCCEEDED),
      "DGifOpenFileName() failed - ",
      error);

  if (DGifSlurp(gifFile) == GIF_ERROR) {
    auto gifFileError = gifFile->Error;
    DGifCloseFile(gifFile, &error);
    TORCH_CHECK(false, "DGifSlurp() failed - ", gifFileError);
  }

  // Note:
  // The GIF format has this notion of "canvas" and "canvas size", where each
  // image could be displayed on the canvas at different offsets, forming a
  // mosaic/picture wall like so:
  //
  // <---    canvas W    --->
  // ------------------------     ^
  // |         |            |     |
  // |   img1  |    img3    |     |
  // |         |------------|  canvas H
  // |----------            |     |
  // |   img2  |    img4    |     |
  // |         |            |     |
  // ------------------------     v
  // The GifLib docs indicate that this is mostly vestigial, and modern viewers
  // ignore the canvas size as well as image offsets. Hence, we're ignoring that
  // too:
  // - We're ignoring the canvas width and height and assume that the shape of
  // the canvas and of all images is the shape of the first image.
  // - We're enforcing that all images have the same shape.
  // - Left and Top offsets of each image are ignored as well and assumed to be
  // 0.

  auto out_h = gifFile->SavedImages[0].ImageDesc.Height;
  auto out_w = gifFile->SavedImages[0].ImageDesc.Width;
  auto num_images = gifFile->ImageCount;

  auto out = torch::empty(
      {int64_t(num_images), 3, int64_t(out_h), int64_t(out_w)}, torch::kU8);
  auto out_a = out.accessor<uint8_t, 4>();

  for (int i = 0; i < num_images; i++) {
    const SavedImage& img = gifFile->SavedImages[i];
    const GifImageDesc& desc = img.ImageDesc;
    TORCH_CHECK(
        desc.Width == out_w && desc.Height == out_h,
        "All images in the gif should have the same dimensions.");

    const ColorMapObject* cmap =
        desc.ColorMap ? desc.ColorMap : gifFile->SColorMap;
    TORCH_CHECK(
        cmap != nullptr,
        "Global and local color maps are missing. This should never happen!");

    for (int h = 0; h < desc.Height; h++) {
      for (int w = 0; w < desc.Width; w++) {
        auto c = img.RasterBits[h * desc.Width + w];
        GifColorType rgb = cmap->Colors[c];
        out_a[i][0][h][w] = rgb.Red;
        out_a[i][1][h][w] = rgb.Green;
        out_a[i][2][h][w] = rgb.Blue;
      }
    }
  }
  out = out.squeeze(0); // remove batch dim if there's only one image

  DGifCloseFile(gifFile, &error);
  TORCH_CHECK(error == D_GIF_SUCCEEDED, "DGifCloseFile() failed - ", error);

  return out;
}

} // namespace image
} // namespace vision
