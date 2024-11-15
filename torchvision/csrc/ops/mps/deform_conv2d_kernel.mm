// vision::ops::
// deform_conv2d_kernal.mm
//

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include "mps_helpers.h"
#include "mps_kernels.h"


namespace vision {
namespace ops {

namespace {

const int64_t tkMaxParallelImgs = 32;


void deformable_im2col(const at::Tensor& input,
                       const at::Tensor& data_offset,
                       const at::Tensor& data_mask,
                       int64_t n_in_channels,
                       int64_t height,
                       int64_t width,
                       int64_t weight_h,
                       int64_t weight_w,
                       int64_t pad_h,
                       int64_t pad_w,
                       int64_t stride_h,
                       int64_t stride_w,
                       int64_t dilation_h,
                       int64_t dilation_w,
                       int64_t out_h,
                       int64_t out_w,
                       int64_t parallel_imgs,
                       int64_t deformable_group,
                       bool use_mask,
                       at::Tensor data_col) {
    using namespace at::native::mps;
    
    // Validate tensors as of type mps.
    TORCH_CHECK(input.is_mps(),         "input must be a MPS tensor");
    TORCH_CHECK(data_offset.is_mps(),   "data_offset must be a MPS tensor");
    TORCH_CHECK(data_mask.is_mps(),     "data_mask must be a MPS tensor");
    
    at::TensorArg input_t{input, "input", 1},
    data_offset_t{data_offset, "data_offset", 2},
    data_mask_t{data_mask, "data_mask", 3};
    
    at::CheckedFrom c = "deformable_im2col";
    at::checkAllSameGPU(c, {input_t, data_offset_t, data_mask_t});
    at::checkAllSameType(c, {input_t, data_offset_t, data_mask_t});
    
    
    const int64_t num_kernels = (int64_t)n_in_channels * out_h * out_w * parallel_imgs;

    // These function parameters have all been made contiguous by the caller function deform_conv2d_forward_kernel
    // Check if it is safe to skip the following:
    auto input_c = input.contiguous();
    auto data_offset_c = data_offset.contiguous();
    auto data_mask_c = data_mask.contiguous();
    
    // Get a raw pointer to the underlying data structure of the tensors and cast it as a pointer to an MTLBuffer.
    id<MTLBuffer> inputBuffer = getMTLBufferStorage(input_c);
    id<MTLBuffer> data_offsetBuffer = getMTLBufferStorage(data_offset_c);
    id<MTLBuffer> data_maskBuffer = getMTLBufferStorage(data_mask_c);
    id<MTLBuffer> data_colBuffer = getMTLBufferStorage(data_col);
  
    id<MTLDevice> device = MPSDevice::getInstance()->device();
        
    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
        @autoreleasepool {
            const std::string kernel = "deformable_im2col_" + scalarToMetalTypeString(input.scalar_type());
            id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

            MTLSize threadgroupsPerGrid = MTLSizeMake(
                std::min(ceil_div(static_cast<int64_t>(num_kernels), 
                static_cast<int64_t>(512)), 
                static_cast<int64_t>(4096)),
                1,
                1);

            // this function call is a no-op if MPS Profiler is not enabled
            getMPSProfiler().beginProfileKernel(visionPSO, kernel, {input_c, data_offset_c, data_mask_c});
            
            id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
            [computeEncoder setComputePipelineState:visionPSO];

            [computeEncoder setBuffer:inputBuffer offset:input_c.storage_offset() * input_c.element_size() atIndex:1];
            [computeEncoder setBuffer:data_offsetBuffer offset:data_offset_c.storage_offset() * data_offset_c.element_size() atIndex:2];
            [computeEncoder setBuffer:data_maskBuffer offset:data_mask_c.storage_offset() * data_mask_c.element_size() atIndex:3];
            [computeEncoder setBuffer:data_colBuffer offset:data_col.storage_offset() * data_col.element_size() atIndex:20];
        
            [computeEncoder setBytes:&num_kernels length:sizeof(int64_t) atIndex:0];
            [computeEncoder setBytes:&height length:sizeof(int64_t) atIndex:4];
            [computeEncoder setBytes:&width length:sizeof(int64_t) atIndex:5];
            [computeEncoder setBytes:&weight_h length:sizeof(int64_t) atIndex:6];
            [computeEncoder setBytes:&weight_w length:sizeof(int64_t) atIndex:7];
            [computeEncoder setBytes:&pad_h length:sizeof(int64_t) atIndex:8];
            [computeEncoder setBytes:&pad_w length:sizeof(int64_t) atIndex:9];
            [computeEncoder setBytes:&stride_h length:sizeof(int64_t) atIndex:10];
            [computeEncoder setBytes:&stride_w length:sizeof(int64_t) atIndex:11];
            [computeEncoder setBytes:&dilation_h length:sizeof(int64_t) atIndex:12];
            [computeEncoder setBytes:&dilation_w length:sizeof(int64_t) atIndex:13];
            [computeEncoder setBytes:&parallel_imgs length:sizeof(int64_t) atIndex:14];
            [computeEncoder setBytes:&n_in_channels length:sizeof(int64_t) atIndex:15];
            [computeEncoder setBytes:&deformable_group length:sizeof(int64_t) atIndex:16];
            [computeEncoder setBytes:&out_h length:sizeof(int64_t) atIndex:17];
            [computeEncoder setBytes:&out_w length:sizeof(int64_t) atIndex:18];
            [computeEncoder setBytes:&use_mask length:sizeof(bool) atIndex:19];
            
            // A threadGroup is equivalent to a cuda's block.
            NSUInteger tgSize = visionPSO.maxTotalThreadsPerThreadgroup;
            if (tgSize > threadsPerBlock) {
                tgSize = threadsPerBlock;
            }
            
            MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
            [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];
            
            getMPSProfiler().endProfileKernel(visionPSO);
        }
    });
    
}

int get_greatest_divisor_below_bound(int n, int bound) {
    for (int k = bound; k > 1; --k) {
        if (n % k == 0) {
            return k;
        }
    }
    return 1;
}

void compute_grad_input(
                        const at::Tensor& columns,
                        const at::Tensor& offset,
                        const at::Tensor& mask,
                        int64_t channels,
                        int64_t height,
                        int64_t width,
                        int64_t weight_h, //kernel_h
                        int64_t weight_w, //kernel_w
                        int64_t pad_h,
                        int64_t pad_w,
                        int64_t stride_h,
                        int64_t stride_w,
                        int64_t dilation_h,
                        int64_t dilation_w,
                        int64_t parallel_imgs, //batch_sz
                        int64_t n_offset_grps,
                        bool use_mask,
                        at::Tensor grad_im) {
    using namespace at::native::mps;
    
    at::globalContext().alertNotDeterministic("compute_grad_input");
    
    auto columns_c = columns.contiguous();
    auto offset_c = offset.contiguous();
    auto mask_c = mask.contiguous();
    
    const int64_t out_h =
    (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
    const int64_t out_w =
    (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
    
    const int64_t num_kernels =
    (int64_t)channels * weight_h * weight_w * out_h * out_w * parallel_imgs;
    
    id<MTLBuffer> columnsBuffer = getMTLBufferStorage(columns_c);
    id<MTLBuffer> offsetBuffer = getMTLBufferStorage(offset_c);
    id<MTLBuffer> maskBuffer = getMTLBufferStorage(mask_c);
    id<MTLBuffer> grad_imBuffer = getMTLBufferStorage(grad_im);
    
    id<MTLDevice> device = MPSDevice::getInstance()->device();
    
    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
            
            const std::string kernel = "deformable_col2im_" + scalarToMetalTypeString(columns.scalar_type());
            id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);
            
            // this function call is a no-op if MPS Profiler is not enabled
            getMPSProfiler().beginProfileKernel(visionPSO, kernel, {columns, offset, mask});
            
            [computeEncoder setComputePipelineState:visionPSO];
            
            [computeEncoder setBuffer:columnsBuffer offset:columns_c.storage_offset() * columns_c.element_size() atIndex:1];
            [computeEncoder setBuffer:offsetBuffer offset:offset_c.storage_offset() * offset_c.element_size() atIndex:2];
            [computeEncoder setBuffer:maskBuffer offset:mask_c.storage_offset() * mask_c.element_size() atIndex:3];
            [computeEncoder setBuffer:grad_imBuffer
                               offset:grad_im.storage_offset() * grad_im.element_size()
                              atIndex:20];
            
            [computeEncoder setBytes:&num_kernels length:sizeof(int64_t) atIndex:0];
            [computeEncoder setBytes:&channels length:sizeof(int64_t) atIndex:4];
            [computeEncoder setBytes:&height length:sizeof(int64_t) atIndex:5];
            [computeEncoder setBytes:&width length:sizeof(int64_t) atIndex:6];
            [computeEncoder setBytes:&weight_h length:sizeof(int64_t) atIndex:7];
            [computeEncoder setBytes:&weight_w length:sizeof(int64_t) atIndex:8];
            [computeEncoder setBytes:&pad_h length:sizeof(int64_t) atIndex:9];
            [computeEncoder setBytes:&pad_w length:sizeof(int64_t) atIndex:10];
            [computeEncoder setBytes:&stride_h length:sizeof(int64_t) atIndex:11];
            [computeEncoder setBytes:&stride_w length:sizeof(int64_t) atIndex:12];
            [computeEncoder setBytes:&dilation_h length:sizeof(int64_t) atIndex:13];
            [computeEncoder setBytes:&dilation_w length:sizeof(int64_t) atIndex:14];
            [computeEncoder setBytes:&parallel_imgs length:sizeof(int64_t) atIndex:15];
            [computeEncoder setBytes:&n_offset_grps length:sizeof(int64_t) atIndex:16];
            [computeEncoder setBytes:&out_h length:sizeof(int64_t) atIndex:17];
            [computeEncoder setBytes:&out_w length:sizeof(int64_t) atIndex:18];
            [computeEncoder setBytes:&use_mask length:sizeof(bool) atIndex:19];
            
            // A threadGroup is equivalent to a cuda's block.
            NSUInteger tgSize = visionPSO.maxTotalThreadsPerThreadgroup;
            if (tgSize > threadsPerBlock) {
                tgSize = threadsPerBlock;
            }
            
            MTLSize threadgroupsPerGrid = MTLSizeMake(
                                                      std::min(ceil_div(static_cast<int64_t>(num_kernels), static_cast<int64_t>(512)), static_cast<int64_t>(4096)),
                                                      1,
                                                      1);
            
            MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
            [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];
            
            getMPSProfiler().endProfileKernel(visionPSO);
        }
    });
}

void compute_grad_offset_and_mask(
                                  const at::Tensor& columns,
                                  const at::Tensor& input,
                                  const at::Tensor& offset,
                                  const at::Tensor& mask,
                                  int64_t channels,
                                  int64_t height,
                                  int64_t width,
                                  int64_t weight_h,
                                  int64_t weight_w,
                                  int64_t pad_h,
                                  int64_t pad_w,
                                  int64_t stride_h,
                                  int64_t stride_w,
                                  int64_t dilation_h,
                                  int64_t dilation_w,
                                  int64_t parallel_imgs,
                                  int64_t n_offset_grps,
                                  bool use_mask,
                                  at::Tensor grad_offset,
                                  at::Tensor grad_mask) {
    
    using namespace at::native::mps;
    
    auto columns_c = columns; //.contiguous();
    auto input_c = input; //.contiguous();
    auto offset_c = offset; //.contiguous();
    auto mask_c = mask; //.contiguous();
    
    const int64_t out_h =
    (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
    const int64_t out_w =
    (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
    const int64_t num_kernels = (int64_t)out_h * out_w * 2 * weight_h * weight_w *
    n_offset_grps * parallel_imgs;
    
    const int64_t offset_channels = 2 * weight_h * weight_w * n_offset_grps;
    
    id<MTLBuffer> columnsBuffer = getMTLBufferStorage(columns_c);
    id<MTLBuffer> inputBuffer = getMTLBufferStorage(input_c);
    id<MTLBuffer> offsetBuffer = getMTLBufferStorage(offset_c);
    id<MTLBuffer> maskBuffer = getMTLBufferStorage(mask_c);
    id<MTLBuffer> grad_offsetBuffer = getMTLBufferStorage(grad_offset);
    id<MTLBuffer> grad_maskBuffer = getMTLBufferStorage(grad_mask);
    
    id<MTLDevice> device = MPSDevice::getInstance()->device();
    
    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
            MTLSize threadgroupsPerGrid = MTLSizeMake(std::min(ceil_div(static_cast<int64_t>(num_kernels), static_cast<int64_t>(512)), static_cast<int64_t>(4096)), 1, 1);

            const std::string kernel = "deformable_col2im_coord_" + scalarToMetalTypeString(columns.scalar_type());
            id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);
            
            // this function call is a no-op if MPS Profiler is not enabled
            getMPSProfiler().beginProfileKernel(visionPSO, kernel, {columns_c, input_c, offset_c, mask_c});
                        
            [computeEncoder setComputePipelineState:visionPSO];
            
            [computeEncoder setBuffer:columnsBuffer offset:columns_c.storage_offset() * columns_c.element_size() atIndex:1];
            [computeEncoder setBuffer:inputBuffer offset:input_c.storage_offset() * input_c.element_size() atIndex:2];
            [computeEncoder setBuffer:offsetBuffer offset:offset_c.storage_offset() * offset_c.element_size() atIndex:3];
            [computeEncoder setBuffer:maskBuffer offset:mask_c.storage_offset() * mask_c.element_size() atIndex:4];
            [computeEncoder setBuffer:grad_offsetBuffer
                               offset:grad_offset.storage_offset() * grad_offset.element_size()
                              atIndex:22];
            [computeEncoder setBuffer:grad_maskBuffer
                               offset:grad_mask.storage_offset() * grad_mask.element_size()
                              atIndex:23];
            
            [computeEncoder setBytes:&num_kernels length:sizeof(int64_t) atIndex:0];
            [computeEncoder setBytes:&channels length:sizeof(int64_t) atIndex:5];
            [computeEncoder setBytes:&height length:sizeof(int64_t) atIndex:6];
            [computeEncoder setBytes:&width length:sizeof(int64_t) atIndex:7];
            [computeEncoder setBytes:&weight_h length:sizeof(int64_t) atIndex:8];
            [computeEncoder setBytes:&weight_w length:sizeof(int64_t) atIndex:9];
            [computeEncoder setBytes:&pad_h length:sizeof(int64_t) atIndex:10];
            [computeEncoder setBytes:&pad_w length:sizeof(int64_t) atIndex:11];
            [computeEncoder setBytes:&stride_h length:sizeof(int64_t) atIndex:12];
            [computeEncoder setBytes:&stride_w length:sizeof(int64_t) atIndex:13];
            [computeEncoder setBytes:&dilation_h length:sizeof(int64_t) atIndex:14];
            [computeEncoder setBytes:&dilation_w length:sizeof(int64_t) atIndex:15];
            [computeEncoder setBytes:&parallel_imgs length:sizeof(int64_t) atIndex:16];
            [computeEncoder setBytes:&offset_channels length:sizeof(int64_t) atIndex:17];
            [computeEncoder setBytes:&n_offset_grps length:sizeof(int64_t) atIndex:18];
            [computeEncoder setBytes:&out_h length:sizeof(int64_t) atIndex:19];
            [computeEncoder setBytes:&out_w length:sizeof(int64_t) atIndex:20];
            [computeEncoder setBytes:&use_mask length:sizeof(bool) atIndex:21];
            
            // A threadGroup is equivalent to a cuda's block.
            NSUInteger tgSize = visionPSO.maxTotalThreadsPerThreadgroup;
            if (tgSize > threadsPerBlock) {
                tgSize = threadsPerBlock;
            }
            
                        
            MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
            [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];
            
            getMPSProfiler().endProfileKernel(visionPSO);
        }
    });    
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_gradient_inputs(
                                                                        at::Tensor input,
                                                                        at::Tensor weight,
                                                                        at::Tensor offset,
                                                                        at::Tensor mask,
                                                                        at::Tensor grad_out,
                                                                        int64_t stride_h,
                                                                        int64_t stride_w,
                                                                        int64_t pad_h,
                                                                        int64_t pad_w,
                                                                        int64_t dilation_h,
                                                                        int64_t dilation_w,
                                                                        int64_t n_weight_grps,
                                                                        int64_t n_offset_grps,
                                                                        int64_t n_parallel_imgs,
                                                                        bool use_mask) {
    
    int64_t batch_sz = input.size(0);
    int64_t n_in_channels = input.size(1);
    int64_t in_h = input.size(2);
    int64_t in_w = input.size(3);
    
    n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);
    
    int64_t n_out_channels = weight.size(0);
    int64_t weight_h = weight.size(2);
    int64_t weight_w = weight.size(3);
    
    int64_t out_w =
    (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
    int64_t out_h =
    (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
    
    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);
    
    if (batch_sz == 0) {
        return std::make_tuple(grad_input, grad_offset, grad_mask);
    }
    
    auto columns = at::empty(
                             {n_in_channels * weight_w * weight_h, n_parallel_imgs * out_h * out_w},
                             input.options());
    
    // Separate into blocks
    grad_input = grad_input.reshape(
                                    {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});
    input = input.reshape(
                          {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});
    
    grad_offset = grad_offset.reshape(
                                      {batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          n_offset_grps * 2 * weight_h * weight_w,
                                          out_h,
                                          out_w});
    offset = offset.reshape(
                            {batch_sz / n_parallel_imgs,
                                n_parallel_imgs,
                                n_offset_grps * 2 * weight_h * weight_w,
                                out_h,
                                out_w});
    
    if (use_mask) {
        grad_mask = grad_mask.reshape(
                                      {batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          n_offset_grps * weight_h * weight_w,
                                          out_h,
                                          out_w});
        mask = mask.reshape(
                            {batch_sz / n_parallel_imgs,
                                n_parallel_imgs,
                                n_offset_grps * weight_h * weight_w,
                                out_h,
                                out_w});
    }
    
    grad_out = grad_out
        .reshape(
                 {batch_sz / n_parallel_imgs,
                     n_parallel_imgs,
                     n_weight_grps,
                     n_out_channels / n_weight_grps,
                     out_h,
                     out_w})
        .permute({0, 2, 3, 1, 4, 5});
    
    weight = weight.reshape(
                            {n_weight_grps,
                                weight.size(0) / n_weight_grps,
                                weight.size(1),
                                weight.size(2),
                                weight.size(3)});
    
    columns = columns.view(
                           {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int64_t elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
        columns.zero_();
        // Separate into weight groups
        for (int64_t g = 0; g < n_weight_grps; g++) {
            columns[g] = columns[g].addmm_(
                                           weight[g].flatten(1).transpose(0, 1), grad_out[elt][g].flatten(1));
        }
        
        compute_grad_offset_and_mask(
                                     columns,
                                     input[elt],
                                     offset[elt],
                                     mask[elt],
                                     n_in_channels,
                                     in_h,
                                     in_w,
                                     weight_h,
                                     weight_w,
                                     pad_h,
                                     pad_w,
                                     stride_h,
                                     stride_w,
                                     dilation_h,
                                     dilation_w,
                                     n_parallel_imgs,
                                     n_offset_grps,
                                     use_mask,
                                     grad_offset[elt],
                                     grad_mask[elt]);
        
        compute_grad_input(
                           columns,
                           offset[elt],
                           mask[elt],
                           n_in_channels,
                           in_h,
                           in_w,
                           weight_h,
                           weight_w,
                           pad_h,
                           pad_w,
                           stride_h,
                           stride_w,
                           dilation_h,
                           dilation_w,
                           n_parallel_imgs,
                           n_offset_grps,
                           use_mask,
                           grad_input[elt]);
    }
    
    grad_input = grad_input.view({batch_sz, n_in_channels, in_h, in_w});
    grad_offset = grad_offset.view(
                                   {batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});
    
    if (use_mask) {
        grad_mask = grad_mask.view(
                                   {batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w});
    }
    
    return std::make_tuple(grad_input, grad_offset, grad_mask);
}

at::Tensor backward_gradient_parameters(
                                        at::Tensor input,
                                        const at::Tensor& weight,
                                        at::Tensor offset,
                                        at::Tensor mask,
                                        const at::Tensor& grad_out,
                                        int64_t stride_h,
                                        int64_t stride_w,
                                        int64_t pad_h,
                                        int64_t pad_w,
                                        int64_t dilation_h,
                                        int64_t dilation_w,
                                        int64_t n_weight_grps,
                                        int64_t n_offset_grps,
                                        int64_t n_parallel_imgs,
                                        bool use_mask) {
    
    int64_t batch_sz = input.size(0);
    int64_t n_in_channels = input.size(1);
    int64_t in_h = input.size(2);
    int64_t in_w = input.size(3);
    
    n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);
    
    int64_t n_out_channels = weight.size(0);
    int64_t weight_h = weight.size(2);
    int64_t weight_w = weight.size(3);
    
    int64_t out_h = grad_out.size(2);
    int64_t out_w = grad_out.size(3);
    
    auto grad_weight = at::zeros_like(weight);
    if (batch_sz == 0) {
        return grad_weight;
    }
    
    at::Tensor grad_out_buf = grad_out
        .reshape(
                 {batch_sz / n_parallel_imgs,
                     n_parallel_imgs,
                     n_weight_grps,
                     n_out_channels / n_weight_grps,
                     out_h,
                     out_w})
        .permute({0, 2, 3, 1, 4, 5})
        .contiguous();
    
    input = input.reshape(
                          {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});
    
    offset = offset.reshape(
                            {batch_sz / n_parallel_imgs,
                                n_parallel_imgs,
                                n_offset_grps * 2 * weight_h * weight_w,
                                out_h,
                                out_w});
    
    if (use_mask) {
        mask = mask.reshape(
                            {batch_sz / n_parallel_imgs,
                                n_parallel_imgs,
                                n_offset_grps * weight_h * weight_w,
                                out_h,
                                out_w});
    }
    
    grad_weight = grad_weight.reshape(
                                      {n_weight_grps,
                                          grad_weight.size(0) / n_weight_grps,
                                          grad_weight.size(1),
                                          grad_weight.size(2),
                                          grad_weight.size(3)});
    
    auto columns = at::empty(
                             {n_weight_grps,
                                 n_in_channels * weight_w * weight_h / n_weight_grps,
                                 n_parallel_imgs * out_h * out_w},
                             input.options());
    
    for (int64_t elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
        deformable_im2col(
                          input[elt],
                          offset[elt],
                          mask[elt],
                          n_in_channels,
                          in_h,
                          in_w,
                          weight_h,
                          weight_w,
                          pad_h,
                          pad_w,
                          stride_h,
                          stride_w,
                          dilation_h,
                          dilation_w,
                          out_h,
                          out_w,
                          n_parallel_imgs,
                          n_offset_grps,
                          use_mask,
                          columns);
        
        for (int64_t g = 0; g < n_weight_grps; g++) {
            grad_weight[g] =
            grad_weight[g]
                .flatten(1)
                .addmm_(
                        grad_out_buf[elt][g].flatten(1), columns[g].transpose(1, 0))
                .view_as(grad_weight[g]);
        }
    }
    
    grad_weight = grad_weight.view(
                                   {grad_weight.size(0) * grad_weight.size(1),
                                       grad_weight.size(2),
                                       grad_weight.size(3),
                                       grad_weight.size(4)});
    return grad_weight;
}

at::Tensor deform_conv2d_forward_kernel(
                                        const at::Tensor& input,
                                        const at::Tensor& weight,
                                        const at::Tensor& offset,
                                        const at::Tensor& mask,
                                        const at::Tensor& bias,
                                        int64_t stride_h,
                                        int64_t stride_w,
                                        int64_t pad_h,
                                        int64_t pad_w,
                                        int64_t dilation_h,
                                        int64_t dilation_w,
                                        int64_t n_weight_grps,
                                        int64_t n_offset_grps,
                                        bool use_mask) {
                                            
    
    at::Tensor input_c = input.contiguous();
    at::Tensor offset_c = offset.contiguous();
    at::Tensor weight_c = weight.contiguous();
    at::Tensor mask_c = mask.contiguous();
    at::Tensor bias_c = bias.contiguous();
    
    TORCH_CHECK(input_c.ndimension() == 4);
    TORCH_CHECK(offset_c.ndimension() == 4);
    TORCH_CHECK(!use_mask || mask_c.ndimension() == 4);
    TORCH_CHECK(weight_c.ndimension() == 4);
    TORCH_CHECK(input_c.is_mps(), "input must be a MPS tensor");
    
    at::DeviceGuard guard(input_c.device());

    int batch_sz = input_c.size(0);
    int in_channels = input_c.size(1);
    int in_h = input_c.size(2);
    int in_w = input_c.size(3);
    
    int n_parallel_imgs =
    get_greatest_divisor_below_bound(batch_sz, tkMaxParallelImgs);

    int out_channels = weight_c.size(0);
    int weight_h = weight_c.size(2);
    int weight_w = weight_c.size(3);
    
    int ker_h = dilation_h * (weight_h - 1) + 1;
    int ker_w = dilation_w * (weight_w - 1) + 1;
    int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
    int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;
    
    TORCH_CHECK(
                weight_h > 0 && weight_w > 0,
                "weight_h: ",
                weight_h,
                " weight_w: ",
                weight_w);
    TORCH_CHECK(
                stride_h > 0 && stride_w > 0,
                "stride_h: ",
                stride_h,
                " stride_w: ",
                stride_w);
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_h, " pad_w: ", pad_w);
    TORCH_CHECK(
                dilation_h > 0 && dilation_w > 0,
                "dilation_h: ",
                dilation_h,
                " dilation_w: ",
                dilation_w);
    
    TORCH_CHECK(weight_c.size(1) * n_weight_grps == input_c.size(1));
    TORCH_CHECK(weight_c.size(0) % n_weight_grps == 0);
    TORCH_CHECK(
                (offset_c.size(1) == n_offset_grps * 2 * weight_h * weight_w),
                "offset.shape[1] is not valid: got: ",
                offset_c.size(1),
                " expected: ",
                n_offset_grps * 2 * weight_h * weight_w);
    TORCH_CHECK(
                (!use_mask || mask_c.size(1) == n_offset_grps * weight_h * weight_w),
                "mask.shape[1] is not valid: got: ",
                mask_c.size(1),
                " expected: ",
                n_offset_grps * weight_h * weight_w);
    TORCH_CHECK(input_c.size(1) % n_offset_grps == 0);
    
    TORCH_CHECK(
                (offset_c.size(0) == input_c.size(0)), "invalid batch size of offset");
    TORCH_CHECK(
                (offset_c.size(2) == out_h && offset_c.size(3) == out_w),
                "offset output dims: (",
                offset_c.size(2),
                ", ",
                offset_c.size(3),
                ") - ",
                "computed output dims: (",
                out_h,
                ", ",
                out_w,
                ")");
    TORCH_CHECK(
                (mask_c.size(0) == input_c.size(0)), "invalid batch size of mask");
    TORCH_CHECK(
                (!use_mask || (mask_c.size(2) == out_h && mask_c.size(3) == out_w)),
                "mask output dims: (",
                mask_c.size(2),
                ", ",
                mask_c.size(3),
                ") - ",
                "computed output dims: (",
                out_h,
                ", ",
                out_w,
                ")");
    TORCH_CHECK(
                out_h > 0 && out_w > 0,
                "Calculated output size too small - out_h: ",
                out_h,
                " out_w: ",
                out_w);
    
    auto out =
    at::zeros({batch_sz, out_channels, out_h, out_w}, input_c.options());
    if (batch_sz == 0) {
        return out;
    }
    
    // Separate batches into blocks
    out = out.view(
                   {batch_sz / n_parallel_imgs,
                       n_parallel_imgs,
                       out_channels,
                       out_h,
                       out_w});
    input_c = input_c.view(
                           {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_h, in_w});
    
    offset_c = offset_c.view(
                             {batch_sz / n_parallel_imgs,
                                 n_parallel_imgs,
                                 n_offset_grps * 2 * weight_h * weight_w,
                                 out_h,
                                 out_w});
    
    if (use_mask) {
        mask_c = mask_c.view(
                             {batch_sz / n_parallel_imgs,
                                 n_parallel_imgs,
                                 n_offset_grps * weight_h * weight_w,
                                 out_h,
                                 out_w});
    }
    
    at::Tensor out_buf = at::zeros(
                                   {batch_sz / n_parallel_imgs,
                                    out_channels,
                                    n_parallel_imgs * out_h,
                                    out_w},
                                   out.options());
    
    // Separate channels into convolution groups
    out_buf = out_buf.view(
                           {out_buf.size(0),
                               n_weight_grps,
                               out_buf.size(1) / n_weight_grps,
                               out_buf.size(2),
                               out_buf.size(3)});
    weight_c = weight_c.view(
                             {n_weight_grps,
                                 weight_c.size(0) / n_weight_grps,
                                 weight_c.size(1),
                                 weight_c.size(2),
                                 weight_c.size(3)});
    
    // Sample points and perform convolution
    auto columns = at::zeros(
                             {in_channels * weight_h * weight_w, n_parallel_imgs * out_h * out_w},
                             input_c.options());
   
    for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
        deformable_im2col(
                          input_c[b],
                          offset_c[b],
                          mask_c[b],
                          in_channels,
                          in_h,
                          in_w,
                          weight_h,
                          weight_w,
                          pad_h,
                          pad_w,
                          stride_h,
                          stride_w,
                          dilation_h,
                          dilation_w,
                          out_h,
                          out_w,
                          n_parallel_imgs,
                          n_offset_grps,
                          use_mask,
                          columns);

        columns = columns.view(
                               {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
        for (int g = 0; g < n_weight_grps; g++) {
            out_buf[b][g] = out_buf[b][g]
                .flatten(1)
                .addmm_(weight_c[g].flatten(1), columns[g])
                .view_as(out_buf[b][g]);
        }
        columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    }
    
    out_buf = out_buf.view(
                           {batch_sz / n_parallel_imgs,
                               out_channels,
                               n_parallel_imgs,
                               out_h,
                               out_w});
    out_buf.transpose_(1, 2);
    out.copy_(out_buf);
    out = out.view({batch_sz, out_channels, out_h, out_w});

    return out + bias_c.view({1, out_channels, 1, 1});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv2d_backward_kernel(
                              const at::Tensor& grad_out,
                              const at::Tensor& input,
                              const at::Tensor& weight,
                              const at::Tensor& offset,
                              const at::Tensor& mask,
                              const at::Tensor& bias,
                              int64_t stride_h,
                              int64_t stride_w,
                              int64_t pad_h,
                              int64_t pad_w,
                              int64_t dilation_h,
                              int64_t dilation_w,
                              int64_t n_weight_grps,
                              int64_t n_offset_grps,
                              bool use_mask) {
    at::Tensor grad_out_c = grad_out.contiguous();
    at::Tensor input_c = input.contiguous();
    at::Tensor weight_c = weight.contiguous();
    at::Tensor offset_c = offset.contiguous();
    at::Tensor mask_c = mask.contiguous();
    at::Tensor bias_c = bias.contiguous();
    
    const int64_t batch_sz = input_c.size(0);
    const int64_t n_parallel_imgs =
    get_greatest_divisor_below_bound(batch_sz, tkMaxParallelImgs);
    
    auto grad_input_and_offset_and_mask = backward_gradient_inputs(
                                                                   input_c,
                                                                   weight_c,
                                                                   offset_c,
                                                                   mask_c,
                                                                   grad_out_c,
                                                                   stride_h,
                                                                   stride_w,
                                                                   pad_h,
                                                                   pad_w,
                                                                   dilation_h,
                                                                   dilation_w,
                                                                   n_weight_grps,
                                                                   n_offset_grps,
                                                                   n_parallel_imgs,
                                                                   use_mask);
    
    auto grad_input = std::get<0>(grad_input_and_offset_and_mask);
    auto grad_offset = std::get<1>(grad_input_and_offset_and_mask);
    auto grad_mask = std::get<2>(grad_input_and_offset_and_mask);
    
    auto grad_weight = backward_gradient_parameters(
                                                    input_c,
                                                    weight_c,
                                                    offset_c,
                                                    mask_c,
                                                    grad_out_c,
                                                    stride_h,
                                                    stride_w,
                                                    pad_h,
                                                    pad_w,
                                                    dilation_h,
                                                    dilation_w,
                                                    n_weight_grps,
                                                    n_offset_grps,
                                                    n_parallel_imgs,
                                                    use_mask);
    
    auto value = grad_out_c.sum({0, 2, 3});
    auto grad_bias = at::ones_like(bias_c) * value;
    
    return std::make_tuple(
                           grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
}
} // namespace


TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
    m.impl(
           TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
           TORCH_FN(deform_conv2d_forward_kernel));
    m.impl(
           TORCH_SELECTIVE_NAME("torchvision::_deform_conv2d_backward"),
           TORCH_FN(deform_conv2d_backward_kernel));
}

} // namespace ops
} // namespace vision

