#ifndef MNASNET_H
#define MNASNET_H

#include <torch/torch.h>
#include "general.h"

namespace vision
{
namespace models
{
struct VISION_API MNASNetImpl : torch::nn::Module
{
    torch::nn::Sequential layers, classifier;

    void _initialize_weights();

public:
    MNASNetImpl(double alpha, double dropout = .2, int64_t num_classes = 1000);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MNASNet);

}  // namespace models
}  // namespace vision

#endif  // MNASNET_H
