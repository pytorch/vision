#pragma once

#include <sys/stat.h>
#include <torch/types.h>

C10_EXPORT torch::Tensor read_file(const std::string& filename);

C10_EXPORT void write_file(const std::string& filename, torch::Tensor& data);
