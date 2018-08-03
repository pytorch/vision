#include <ATen/ATen.h>
#include <TH/TH.h>

std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor &input,
                                                        const at::Tensor &rois,
                                                        const float spatial_scale,
                                                        const int pooled_height,
                                                        const int pooled_width)
{
    AT_ASSERTM(input.type().is_cpu(), "input must be a CPU tensor");
    AT_ASSERTM(rois.type().is_cpu(), "rois must be a CPU tensor");

    auto num_rois = rois.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    at::Tensor output = input.type().tensor({num_rois, channels, pooled_height, pooled_width});
    at::Tensor argmax = input.type().toScalarType(at::kInt).tensor({num_rois, channels, pooled_height, pooled_width}).zero_();

    auto output_size = num_rois * pooled_height * pooled_width * channels;

    if (output.numel() == 0)
    {
        return std::make_tuple(output, argmax);
    }

    /**
     * 
     * Need to perform actual ROI Pooling here
     * 
     **/
    return std::make_tuple(output, argmax);
}