#pragma once

#include <errno.h>
#include <sys/stat.h>
#include <torch/torch.h>

C10_EXPORT torch::Tensor read_file(std::string filename);
