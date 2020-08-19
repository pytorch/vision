#ifndef REGISTER_H
#define REGISTER_H

#include "Video.h"


    m.class_<Video>("video")
        .def(torch::init<std::string, std::string, bool>())
        .def("get_metadata", &Video::getMetadata);
}
#endif
