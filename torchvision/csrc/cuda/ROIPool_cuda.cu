#include <ATen/ATen.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void RoIPoolFForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T* bottom_rois, T* top_data, int* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height)
                       / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width)
                       / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (offset_bottom_data[bottom_index] > maxval) {
          maxval = offset_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename T>
__global__ void RoIPoolFBackward(const int nthreads, const T* top_diff,
    const int* argmax_data, const int num_rois, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    T* offset_bottom_diff = bottom_diff + bottom_offset;
    const int* offset_argmax_data = argmax_data + top_offset;

    int argmax = offset_argmax_data[ph * pooled_width + pw];
    if (argmax != -1) {
      atomicAdd(
          offset_bottom_diff + argmax,
          static_cast<T>(offset_top_diff[ph * pooled_width + pw]));

    }
  }
}

std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width) {
  AT_ASSERT(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERT(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = input.type().tensor({num_rois, channels, pooled_height, pooled_width});
  at::Tensor argmax = input.type().toScalarType(at::kInt).tensor({num_rois, channels, pooled_height, pooled_width}).zero_();

  auto output_size = num_rois * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(output, argmax);
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIPool_forward", [&] {
    RoIPoolFForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.data<scalar_t>(),
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         rois.data<scalar_t>(),
         output.data<scalar_t>(),
         argmax.data<int>());
  });
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, argmax);
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  AT_ASSERT(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERT(rois.type().is_cuda(), "rois must be a CUDA tensor");
  // TODO add more checks

  auto num_rois = rois.size(0);
  at::Tensor grad_input = grad.type().tensor({batch_size, channels, height, width}).zero_();

  cudaStream_t stream = at::globalContext().getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "ROIPool_backward", [&] {
    RoIPoolFBackward<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.data<scalar_t>(),
         argmax.data<int>(),
         num_rois,
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         grad_input.data<scalar_t>(),
         rois.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
