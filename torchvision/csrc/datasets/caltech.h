#ifndef CALTECH_H
#define CALTECH_H

#include <torch/torch.h>

struct Caltech101 : torch::data::Dataset<Caltech101>
{
    Caltech101();
};

#endif  // CALTECH_H
