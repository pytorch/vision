#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include "vision_kernels.h"
#include "mps_helpers.h"

#include <iostream>
#include <cmath>

namespace vision {
namespace ops {

namespace {

// This should be in sync with the one in metal kernel.
int const threadsPerBlock = 512;

at::Tensor roi_align_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {

  using namespace at::native::mps;
  TORCH_CHECK(input.is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(rois.is_mps(), "rois must be a MPS tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});
  
  int64_t num_rois = rois.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  float spatial_scale_f = static_cast<float>(spatial_scale);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  
  int64_t output_size = num_rois * pooled_height * pooled_width * channels;

  if (output.numel() == 0) {
    return output;
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();

  id<MTLBuffer> inputBuffer = getMTLBufferStorage(input_);
  id<MTLBuffer> roisBuffer = getMTLBufferStorage(rois_);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize threadgroupsPerGrid = MTLSizeMake(std::min(ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)), static_cast<int64_t>(4096)), 1, 1);

      const std::string kernel = "roi_align_" + scalarToMetalTypeString(input.scalar_type());
      id<MTLComputePipelineState> binaryPSO = mps::binaryPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel, {input_, rois_});

      [computeEncoder setComputePipelineState:binaryPSO];
      // [N, C, H, W]
      [computeEncoder setBuffer:inputBuffer offset:input_.storage_offset() * input_.element_size() atIndex:0];
      [computeEncoder setBuffer:roisBuffer offset:rois_.storage_offset() * rois_.element_size() atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:output.storage_offset() * output.element_size() atIndex:2];
      
      [computeEncoder setBytes:&output_size length:sizeof(int64_t) atIndex:3];
      [computeEncoder setBytes:&channels length:sizeof(int64_t) atIndex:4];
      [computeEncoder setBytes:&height length:sizeof(int64_t) atIndex:5];
      [computeEncoder setBytes:&width length:sizeof(int64_t) atIndex:6];
      [computeEncoder setBytes:&pooled_height length:sizeof(int64_t) atIndex:7];
      [computeEncoder setBytes:&pooled_width length:sizeof(int64_t) atIndex:8];
      [computeEncoder setBytes:&sampling_ratio length:sizeof(int64_t) atIndex:9];
      [computeEncoder setBytes:&aligned length:sizeof(bool) atIndex:10];
      [computeEncoder setBytes:&spatial_scale_f length:sizeof(float) atIndex:11];

      // A threadGroup is equivalent to a cuda's block.
      NSUInteger tgSize = binaryPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > threadsPerBlock) {
        tgSize = threadsPerBlock;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
  return output;
}

at::Tensor roi_align_backward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {

  using namespace at::native::mps;
  TORCH_CHECK(input.is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(rois.is_mps(), "rois must be a MPS tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});
  
  int64_t num_rois = rois.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  float spatial_scale_f = static_cast<float>(spatial_scale);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  
  int64_t output_size = num_rois * pooled_height * pooled_width * channels;

  if (output.numel() == 0) {
    return output;
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();

  id<MTLBuffer> inputBuffer = getMTLBufferStorage(input_);
  id<MTLBuffer> roisBuffer = getMTLBufferStorage(rois_);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize threadgroupsPerGrid = MTLSizeMake(std::min(ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)), static_cast<int64_t>(4096)), 1, 1);

      const std::string kernel = "roi_align_" + scalarToMetalTypeString(input.scalar_type());
      id<MTLComputePipelineState> binaryPSO = mps::binaryPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel, {input_, rois_});

      [computeEncoder setComputePipelineState:binaryPSO];
      // [N, C, H, W]
      [computeEncoder setBuffer:inputBuffer offset:input_.storage_offset() * input_.element_size() atIndex:0];
      [computeEncoder setBuffer:roisBuffer offset:rois_.storage_offset() * rois_.element_size() atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:output.storage_offset() * output.element_size() atIndex:2];
      
      [computeEncoder setBytes:&output_size length:sizeof(int64_t) atIndex:3];
      [computeEncoder setBytes:&channels length:sizeof(int64_t) atIndex:4];
      [computeEncoder setBytes:&height length:sizeof(int64_t) atIndex:5];
      [computeEncoder setBytes:&width length:sizeof(int64_t) atIndex:6];
      [computeEncoder setBytes:&pooled_height length:sizeof(int64_t) atIndex:7];
      [computeEncoder setBytes:&pooled_width length:sizeof(int64_t) atIndex:8];
      [computeEncoder setBytes:&sampling_ratio length:sizeof(int64_t) atIndex:9];
      [computeEncoder setBytes:&aligned length:sizeof(bool) atIndex:10];
      [computeEncoder setBytes:&spatial_scale_f length:sizeof(float) atIndex:11];

      // A threadGroup is equivalent to a cuda's block.
      NSUInteger tgSize = binaryPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > threadsPerBlock) {
        tgSize = threadsPerBlock;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN(roi_align_forward_kernel));
  //m.impl(
  //    TORCH_SELECTIVE_NAME("torchvision::_roi_align_backward"),
  //    TORCH_FN(roi_align_backward_kernel));
}

} // namespace ops
} // namespace vision
