#include "decode_gif.h"
#include <stdio.h>  // TODO: Remove
#include <iostream>  // TODO: Remove?


#include <gif_lib.h>

namespace vision {
namespace image {

torch::Tensor decode_gif(const std::string& path) {
    printf("In decode_gif()!!\n");
    std::cout << "path=" << path << std::endl;

    int error = D_GIF_SUCCEEDED;
    GifFileType* gifFile = DGifOpenFileName(path.c_str(), &error);
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
    std::cout << path << ": " << gifFile->SWidth << "x" << gifFile->SHeight << std::endl;
    std::cout << "Number of images: " << gifFile->ImageCount << std::endl;

    const SavedImage& saved = gifFile->SavedImages[0];  // TODO: handle multple images?
    const GifImageDesc& desc = saved.ImageDesc;
    const ColorMapObject* colorMap = desc.ColorMap ? desc.ColorMap : commonMap;
    // TODO What are Left and Top used for?
    std::cout << desc.Width << "x" << desc.Height << "+" << desc.Left << "," << desc.Top
        << ", has local colorMap: " << (desc.ColorMap ? "Yes" : "No") << std::endl;

    auto out = torch::empty({3, int64_t(desc.Height), int64_t(desc.Width)}, torch::kU8);

    std::stringstream ss;
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