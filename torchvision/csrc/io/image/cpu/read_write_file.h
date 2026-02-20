#pragma once

#include "../../../StableABICompat.h"

namespace vision {
namespace image {

vision::stable::Tensor read_file(std::string filename);

void write_file(std::string filename, const vision::stable::Tensor& data);

} // namespace image
} // namespace vision
