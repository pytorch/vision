
#include "image.h"
#include <ATen/ATen.h>

static auto registry = torch::RegisterOperators()
                           .op("image::decode_png", &decodePNG)
                           .op("image::encode_png", &encodePNG)
                           .op("image::decode_jpeg", &decodeJPEG)
                           .op("image::encode_jpeg", &encodeJPEG)
                           .op("image::read_file", &read_file)
                           .op("image::write_file", &write_file)
                           .op("image::decode_image", &decode_image);
