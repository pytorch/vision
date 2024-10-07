#include "decode_gif.h"
#include <cstring>
#include "giflib/gif_lib.h"

namespace vision {
namespace image {

typedef struct reader_helper_t {
  uint8_t const* encoded_data; // input tensor data pointer
  size_t encoded_data_size; // size of input tensor in bytes
  size_t num_bytes_read; // number of bytes read so far in the tensor
} reader_helper_t;

// That function is used by GIFLIB routines to read the encoded bytes.
// This reads `len` bytes and writes them into `buf`. The data is read from the
// input tensor passed to decode_gif() starting at the `num_bytes_read`
// position.
int read_from_tensor(GifFileType* gifFile, GifByteType* buf, int len) {
  // the UserData field was set in DGifOpen()
  reader_helper_t* reader_helper =
      static_cast<reader_helper_t*>(gifFile->UserData);

  size_t num_bytes_to_read = std::min(
      (size_t)len,
      reader_helper->encoded_data_size - reader_helper->num_bytes_read);
  std::memcpy(
      buf, reader_helper->encoded_data + reader_helper->num_bytes_read, len);
  reader_helper->num_bytes_read += num_bytes_to_read;
  return num_bytes_to_read;
}

torch::Tensor decode_gif(const torch::Tensor& encoded_data) {
  // LibGif docs: https://giflib.sourceforge.net/intro.html
  // Refer over there for more details on the libgif API, API ref, and a
  // detailed description of the GIF format.

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
      DGifOpen(static_cast<void*>(&reader_helper), read_from_tensor, &error);

  TORCH_CHECK(
      (gifFile != nullptr) && (error == D_GIF_SUCCEEDED),
      "DGifOpenFileName() failed - ",
      error);

  if (DGifSlurp(gifFile) == GIF_ERROR) {
    auto gifFileError = gifFile->Error;
    DGifCloseFile(gifFile, &error);
    TORCH_CHECK(false, "DGifSlurp() failed - ", gifFileError);
  }
  auto num_images = gifFile->ImageCount;

  // This check should already done within DGifSlurp(), just to be safe
  TORCH_CHECK(num_images > 0, "GIF file should contain at least one image!");

  GifColorType bg = {0, 0, 0};
  if (gifFile->SColorMap) {
    bg = gifFile->SColorMap->Colors[gifFile->SBackGroundColor];
  }

  // The GIFLIB docs say that the canvas's height and width are potentially
  // ignored by modern viewers, so to be on the safe side we set the output
  // height to max(canvas_heigh, first_image_height). Same for width.
  // https://giflib.sourceforge.net/whatsinagif/bits_and_bytes.html
  auto out_h =
      std::max(gifFile->SHeight, gifFile->SavedImages[0].ImageDesc.Height);
  auto out_w =
      std::max(gifFile->SWidth, gifFile->SavedImages[0].ImageDesc.Width);

  // We output a channels-last tensor for consistency with other image decoders.
  // Torchvision's resize tends to be is faster on uint8 channels-last tensors.
  auto options = torch::TensorOptions()
                     .dtype(torch::kU8)
                     .memory_format(torch::MemoryFormat::ChannelsLast);
  auto out = torch::empty(
      {int64_t(num_images), 3, int64_t(out_h), int64_t(out_w)}, options);
  auto out_a = out.accessor<uint8_t, 4>();
  for (int i = 0; i < num_images; i++) {
    const SavedImage& img = gifFile->SavedImages[i];

    GraphicsControlBlock gcb;
    DGifSavedExtensionToGCB(gifFile, i, &gcb);

    const GifImageDesc& desc = img.ImageDesc;
    const ColorMapObject* cmap =
        desc.ColorMap ? desc.ColorMap : gifFile->SColorMap;
    TORCH_CHECK(
        cmap != nullptr,
        "Global and local color maps are missing. This should never happen!");

    // When going from one image to another, there is a "disposal method" which
    // specifies how to handle the transition. E.g. DISPOSE_DO_NOT means that
    // the current image should essentially be drawn on top of the previous
    // canvas. The pixels of that previous canvas will appear on the new one if
    // either:
    // - a pixel is transparent in the current image
    // - the current image is smaller than the canvas, hence exposing its pixels
    // The "background" disposal method means that the current canvas should be
    // set to the background color.
    // We only support these 2 modes and default to "background" when the
    // disposal method is unspecified, or when it's set to "DISPOSE_PREVIOUS"
    // which according to GIFLIB is not widely supported.
    // (https://giflib.sourceforge.net/whatsinagif/animation_and_transparency.html).
    if (i > 0 && gcb.DisposalMode == DISPOSE_DO_NOT) {
      out[i] = out[i - 1];
    } else {
      // Background. If bg wasn't defined, it will be (0, 0, 0)
      for (int h = 0; h < gifFile->SHeight; h++) {
        for (int w = 0; w < gifFile->SWidth; w++) {
          out_a[i][0][h][w] = bg.Red;
          out_a[i][1][h][w] = bg.Green;
          out_a[i][2][h][w] = bg.Blue;
        }
      }
    }

    for (int h = 0; h < desc.Height; h++) {
      for (int w = 0; w < desc.Width; w++) {
        auto c = img.RasterBits[h * desc.Width + w];
        if (c == gcb.TransparentColor) {
          continue;
        }
        GifColorType rgb = cmap->Colors[c];
        out_a[i][0][h + desc.Top][w + desc.Left] = rgb.Red;
        out_a[i][1][h + desc.Top][w + desc.Left] = rgb.Green;
        out_a[i][2][h + desc.Top][w + desc.Left] = rgb.Blue;
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
