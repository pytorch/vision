#pragma once

#include <errno.h>
#include <sys/stat.h>
#include <torch/torch.h>

#ifdef _WIN32
#ifdef UNICODE
#define VISION_STRING std::wstring
#else
#define VISION_STRING std::string
#endif
#else
#define VISION_STRING std::string
#endif

C10_EXPORT torch::Tensor read_file(VISION_STRING filename);

C10_EXPORT void write_file(VISION_STRING filename, torch::Tensor& data);
