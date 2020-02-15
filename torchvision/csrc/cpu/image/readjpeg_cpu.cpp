#include "readjpeg_cpu.h"

#include <setjmp.h>
#include <string>
#include <turbojpeg.h>

torch::Tensor decodeJPEG(const torch::Tensor& data) {

  tjhandle tjInstance = tjInitDecompress();
  if (tjInstance == NULL) {
    TORCH_CHECK(false, "libjpeg-turbo decompression initialization failed.");
  }

  auto datap = data.accessor<unsigned char, 1>().data();

  int width, height;

  if (tjDecompressHeader(tjInstance, datap, data.numel(), &width, &height) < 0) {
    tjDestroy(tjInstance);
    TORCH_CHECK(false, "Error while reading jpeg headers");
  }
  auto tensor =
      torch::empty({int64_t(height), int64_t(width), int64_t(3)}, torch::kU8);

  auto ptr = tensor.accessor<uint8_t, 3>().data();

  int pixelFormat = TJPF_RGB;

  auto ret = tjDecompress2(tjInstance, datap, data.numel(), ptr, width, 0, height,
                      pixelFormat, NULL);
  if(ret != 0){
      tjDestroy(tjInstance);
      TORCH_CHECK(false, "decompressing JPEG image");
  }

  return tensor;
}
