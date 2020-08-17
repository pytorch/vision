#ifndef REGISTER_H
#define REGISTER_H

#include "Video.h"

namespace {

////////////////////////////////////////////////////////////////////////////////
// typedefs.h
////////////////////////////////////////////////////////////////////////////////
static auto registerVideo =
    torch::class_<Video>("torchvision", "Video")
        .def(torch::init<std::string, std::string, bool, int64_t, int64_t>());

} //namespace
#endif
