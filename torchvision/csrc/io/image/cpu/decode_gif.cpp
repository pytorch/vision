#include "decode_gif.h"
#include <stdio.h>  // TODO: Remove
#include <iostream>  // TODO: Remove?


#include <gif_lib.h>

namespace vision {
namespace image {

typedef struct reader_helper_t {
    torch::Tensor const* encoded_bytes;
    int num_bytes_read;  // number of bytes read so far in the tensor
} reader_helper_t;

// Reader function for libgif. This reads `len` bytes and writes them into
// `buf`. The data is read from the UserData field of the `gif` paramater which
// was set when DGifOpen() was called.
int read_from_tensor(GifFileType* gif, GifByteType* buf, int len) {
    reader_helper_t* reader_helper = (reader_helper_t *)gif->UserData;
    auto num_encoded_bytes = (int)reader_helper->encoded_bytes->numel();
    auto data_ptr = reader_helper->encoded_bytes->data_ptr<GifByteType>();

    auto i = 0;
    auto num_bytes_to_read = std::min(len, num_encoded_bytes - reader_helper->num_bytes_read);
    while (i < num_bytes_to_read) {
        buf[i] = data_ptr[reader_helper->num_bytes_read + i];
        i++;
    }
    reader_helper->num_bytes_read += i;
    return i;
}

torch::Tensor decode_gif(const torch::Tensor& encoded_bytes) {

    int error = D_GIF_SUCCEEDED;

    // The other entrypoints of libgif are DGifOpenFileName and
    // DGifOpenFileHandle but we don't want to use those since we are reading
    // encoded bytes from a tensor (for consistency with existing jpeg and png
    // decoders). Using DGifOpen is the only way to read from a custom source.
    // For that we need to provide a reader function that reads from the tensor,
    // and we keep track of the number of bytes read via the reader_helper
    // struct.
    // Note: it does seem like this is involving an extra copy (which happens in
    // read_from_tensor: from the tensor to `buf`), compared to using
    // DGifOpenFileName() or DGifOpenFileHandle().
    reader_helper_t reader_helper;
    reader_helper.encoded_bytes = &encoded_bytes;
    reader_helper.num_bytes_read = 0;
    GifFileType* gifFile = DGifOpen((void *)&reader_helper, read_from_tensor, &error);

    TORCH_CHECK(
        (gifFile != nullptr) && (error == D_GIF_SUCCEEDED),
        "DGifOpenFileName() failed - ", error
    );

    if (DGifSlurp(gifFile) == GIF_ERROR) {
        auto gifFileError = gifFile->Error;
        DGifCloseFile(gifFile, &error);
        TORCH_CHECK(false, "DGifSlurp() failed - ", gifFileError);
    }

    ColorMapObject* commonMap = gifFile->SColorMap;
    std::cout << ": " << gifFile->SWidth << "x" << gifFile->SHeight << std::endl;
    std::cout << "Number of images: " << gifFile->ImageCount << std::endl;

    const SavedImage& saved = gifFile->SavedImages[0];  // TODO: handle multple images?
    const GifImageDesc& desc = saved.ImageDesc;
    const ColorMapObject* colorMap = desc.ColorMap ? desc.ColorMap : commonMap;
    // TODO What are Left and Top used for?
    std::cout << desc.Width << "x" << desc.Height << "+" << desc.Left << "," << desc.Top
        << ", has local colorMap: " << (desc.ColorMap ? "Yes" : "No") << std::endl;

    auto out = torch::empty({3, int64_t(desc.Height), int64_t(desc.Width)}, torch::kU8);

    for (int h = 0; h < desc.Height; h++) {
        for (int w = 0; w < desc.Width; w++) {
            int c = saved.RasterBits[h * desc.Width + w];
            if (colorMap) { // TODO Can there be no colormap????
                GifColorType rgb = colorMap->Colors[c];
                out.index_put_({0, h, w}, rgb.Red);
                out.index_put_({1, h, w}, rgb.Green);
                out.index_put_({2, h, w}, rgb.Blue);
            }
        }
    }

    DGifCloseFile(gifFile, &error);
    TORCH_CHECK(error == D_GIF_SUCCEEDED, "DGifCloseFile() failed - ", error);

    return out;
}

} // namespace image
} // namespace vision