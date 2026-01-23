#pragma once

#include "cpu/decode_gif.h"
#include "cpu/decode_image.h"
#include "cpu/decode_jpeg.h"
#include "cpu/decode_png.h"
#include "cpu/decode_webp.h"
#include "cpu/encode_jpeg.h"
#include "cpu/encode_png.h"
#include "cpu/read_write_file.h"
// CUDA JPEG is disabled when building with stable ABI (TORCH_TARGET_VERSION)
// because the stable ABI doesn't expose raw CUDA streams needed by nvJPEG.
// #include "cuda/encode_decode_jpegs_cuda.h"
