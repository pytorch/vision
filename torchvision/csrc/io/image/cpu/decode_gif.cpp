#include "decode_gif.h"
#include <stdio.h> // TODO: Remove
#include <iostream> // TODO: Remove?

#include <gif_lib.h>

namespace vision {
namespace image {

typedef struct reader_helper_t {
  torch::Tensor const* encoded_bytes;
  int num_bytes_read; // number of bytes read so far in the tensor
} reader_helper_t;

// Reader function for libgif. This reads `len` bytes and writes them into
// `buf`. The data is read from the UserData field of the `gifFile` paramater
// which was set when DGifOpen() was called.
int read_from_tensor(GifFileType* gifFile, GifByteType* buf, int len) {
  reader_helper_t* reader_helper = (reader_helper_t*)gifFile->UserData;
  auto num_encoded_bytes = (int)reader_helper->encoded_bytes->numel();
  auto data_ptr = reader_helper->encoded_bytes->data_ptr<GifByteType>();

  auto i = 0;
  auto num_bytes_to_read =
      std::min(len, num_encoded_bytes - reader_helper->num_bytes_read);
  while (i < num_bytes_to_read) {
    buf[i] = data_ptr[reader_helper->num_bytes_read + i];
    i++;
  }
  reader_helper->num_bytes_read += i;
  return i;
}

torch::Tensor decode_gif(const torch::Tensor& encoded_bytes) {
  // LibGif docs: https://giflib.sourceforge.net/intro.html
  // Refer over there for more details on the libgif API, API ref, and a
  // detailed description of the GIF format.

  int error = D_GIF_SUCCEEDED;

  // We're using DGidOpen. The other entrypoints of libgif are
  // DGifOpenFileName and DGifOpenFileHandle but we don't want to use those,
  // since we need to read the encoded bytes from a tensor (for consistency
  // with existing jpeg and png decoders). Using DGifOpen is the only way to
  // read from a custom source.
  // For that we need to provide a reader function `read_from_tensor` that
  // reads from the tensor, and we keep track of the number of bytes read so
  // far via the reader_helper struct.
  // Note: it does seem like this is involving an extra copy (which happens in
  // read_from_tensor: from the tensor to `buf`), compared to using
  // DGifOpenFileName() or DGifOpenFileHandle().
  reader_helper_t reader_helper;
  reader_helper.encoded_bytes = &encoded_bytes;
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
  // - We're assuming and enforcing that all images have the same height and
  // width
  // - We're ignoring the canvas width and height and assume shape is that of
  // the first image
  // - We're ignoring the top and left offsets and enforce them to be 0.
  // This is for simplicity. All of those may be revisited in the future if need
  // be.

  // TODO: Transparency???

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
    TORCH_CHECK(
        desc.Top == 0 && desc.Left == 0,
        "Images are expected to have top and left offsets of 0.");

    const ColorMapObject* cmap =
        desc.ColorMap ? desc.ColorMap : gifFile->SColorMap;
    TORCH_CHECK(cmap != nullptr, "ColorMap is missing.");

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