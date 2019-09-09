#include <torch/script.h>

#include "ROIAlign.h"
#include "ROIPool.h"
#include "nms.h"

using namespace at;

static auto registry =
    torch::RegisterOperators()
        .op("torchvision::nms", &nms)
        .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor",
            &ROIAlign_forward)
        .op("torchvision::roi_pool", &ROIPool_forward);
