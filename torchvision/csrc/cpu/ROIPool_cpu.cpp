#include <algorithm>
#include <ATen/ATen.h>
#include <TH/TH.h>

std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cpu(const at::Tensor &input,
                                                       const at::Tensor &rois,
                                                       const float spatial_scale,
                                                       const int pooled_height,
                                                       const int pooled_width)
{
    AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
    AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");

    int num_rois = rois.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    at::Tensor output = input.type().tensor({num_rois, channels, pooled_height, pooled_width});
    at::Tensor argmax = input.type().toScalarType(at::kInt).tensor({num_rois, channels, pooled_height, pooled_width}).zero_();

    // define accessors for indexing
    auto input_a = input.accessor<float, 4>();
    auto rois_a = rois.accessor<float, 2>();
    auto output_a = output.accessor<float, 4>();
    auto argmax_a = argmax.accessor<int, 4>();

    if (output.numel() == 0)
    {
        return std::make_tuple(output, argmax);
    }

    for (int n = 0; n < num_rois; ++n)
    {
        int roi_batch_ind = rois_a[n][0];
        int roi_start_w = round(rois_a[n][1] * spatial_scale);
        int roi_start_h = round(rois_a[n][2] * spatial_scale);
        int roi_end_w = round(rois_a[n][3] * spatial_scale);
        int roi_end_h = round(rois_a[n][4] * spatial_scale);

        // Force malformed ROIs to be 1x1 or HxW
        int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

        for (int ph = 0; ph < pooled_height; ++ph)
        {
            for (int pw = 0; pw < pooled_width; ++pw)
            {
                int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
                int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
                int hend = static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
                int wend = static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

                // Add roi offsets and clip to input boundaries
                hstart = std::min(std::max(hstart + roi_start_h, 0), input_height);
                hend = std::min(std::max(hend + roi_start_h, 0), input_height);
                wstart = std::min(std::max(wstart + roi_start_w, 0), input_width);
                wend = std::min(std::max(wend + roi_start_w, 0), input_width);
                bool is_empty = (hend <= hstart) || (wend <= wstart);

                // Define an empty pooling region to be zero
                float maxval = is_empty ? 0 : -FLT_MAX;
                // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
                int maxidx = -1;

                for (int c = 0; c < channels; ++c)
                {
                    for (int h = hstart; h < hend; ++h)
                    {
                        for (int w = wstart; w < wend; ++w)
                        {
                            int index = h * input_width + w;
                            if (input_a[roi_batch_ind][c][h][w] > maxval)
                            {
                                maxval = input_a[roi_batch_ind][c][h][w];
                                maxidx = index;
                            }
                        }
                    }
                    output_a[n][c][ph][pw] = maxval;
                    argmax_a[n][c][ph][pw] = maxidx;
                }
            }
        }
    }

    return std::make_tuple(output, argmax);
}

at::Tensor ROIPool_backward_cpu(const at::Tensor &grad,
                                const at::Tensor &rois,
                                const at::Tensor &argmax,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int batch_size,
                                const int channels,
                                const int height,
                                const int width)
{
    // Check if input tensors are CPU tensors
    AT_ASSERTM(grad.device().is_cpu(), "grad must be a CPU tensor");
    AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");
    AT_ASSERTM(argmax.device().is_cpu(), "argmax must be a CPU tensor");

    auto num_rois = rois.size(0);

    at::Tensor grad_input = at::zeros({batch_size, channels, height, width}, grad.type());

    // handle possibly empty gradients
    if (grad.numel() == 0)
    {
        return grad_input;
    }

    // get stride values to ensure indexing into gradients is correct.
    int n_stride = grad.stride(0);
    int c_stride = grad.stride(1);
    int h_stride = grad.stride(2);
    int w_stride = grad.stride(3);

    // define accessors for tensors
    auto grad_input_a = grad_input.accessor<float, 4>();
    auto grad_a = grad.accessor<float, 4>();
    auto argmax_a = argmax.accessor<int, 4>();
    auto rois_a = rois.accessor<float, 2>();

    for (int n = 0; n < num_rois; ++n)
    {
        int roi_batch_ind = rois_a[n][0];

        for (int c = 0; c < channels; ++c)
        {
            for (int ph = 0; ph < pooled_height; ++ph)
            {
                for (int pw = 0; pw < pooled_width; ++pw)
                {
                    int argmax_idx = argmax_a[n][c][ph][pw];
                    // get height and width index from argmax index
                    int h = argmax_idx / height;
                    int w = argmax_idx % width;

                    grad_input_a[roi_batch_ind][c][h][w] += grad_a[n * n_stride][c * c_stride][ph * h_stride][pw * w_stride];
                }
            }
        }
    }

    return grad_input;
}