#include "readpng_cpu.h"

#include <png.h>
#include <setjmp.h>
#include <string>
#include <turbojpeg.h>

torch::Tensor decodeJPEG(const torch::Tensor& data) {

  tjhandle tjInstance = tjInitDecompress();
  if (tjInstance == NULL) {
    TORCH_CHECK("libjpeg-turbo decompression initialization failed.");
  }

  auto datap = data.accessor<unsigned char, 1>().data();

  int width, height;
  if (tjDecompressHeader(tjInstance, datap, data.numel(), &width, &height) < 0) {
    tjDestroy(tjInstance);
    TORCH_CHECK("Error while reading jpeg headers");
  }

  auto tensor =
      torch::empty({int64_t(height), int64_t(width), int64_t(3)}, torch::kU8);
  auto ptr = tensor.accessor<uint8_t, 3>().data();

  int pixelFormat = TJPF_RGB;
  if (tjDecompress(tjInstance, datap, data.numel(), ptr, width, 0, height,
                      pixelFormat, 0) < 0){
      tjDestroy(tjInstance);
      TORCH_CHECK("decompressing JPEG image");
  }

  tjDestroy(tjInstance);

  return tensor;
}
