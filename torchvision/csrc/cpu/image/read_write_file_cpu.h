#pragma once

#include <torch/torch.h>
#include <errno.h>
#include <sys/stat.h>

C10_EXPORT torch::Tensor read_file(std::string filename);
