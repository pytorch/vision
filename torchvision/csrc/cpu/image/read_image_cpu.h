#pragma once

#include "readjpeg_cpu.h"
#include "readpng_cpu.h"

C10_EXPORT torch::Tensor decode_image(const torch::Tensor& data);
