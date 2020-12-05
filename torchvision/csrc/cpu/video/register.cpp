#pragma once

#include <ATen/ATen.h>

#include "Frame.h"
#include "Video.h"

namespace {

static auto registerFrame =
    torch::class_<Frame>("torchvision", "Frame")
        .def(torch::init<std::string, double, torch::Tensor>())
        .def("type", &Frame::getType)
        .def("pts", &Frame::getPTS)
        .def("data_t", &Frame::getData);

static auto registerVideo =
    torch::class_<Video>("torchvision", "Video")
        .def(torch::init<std::string, std::string>())
        .def("get_current_stream", &Video::getCurrentStream)
        .def("set_current_stream", &Video::setCurrentStream)
        .def("get_metadata", &Video::getStreamMetadata)
        .def("seek", &Video::Seek)
        .def("next", &Video::Next);

} // namespace
