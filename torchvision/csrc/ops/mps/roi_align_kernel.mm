#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include "mps_helpers.h"
#include "mps_kernels.h"

namespace vision {
namespace ops {

namespace {

at::Tensor roi_align_forward_kernel(const at::Tensor& input,
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

  at::Tensor output = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options());

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
      MTLSize threadgroupsPerGrid = MTLSizeMake(
          std::min(ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)), static_cast<int64_t>(4096)),
          1,
          1);

      const std::string kernel = "roi_align_" + scalarToMetalTypeString(input.scalar_type());
      id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(visionPSO, kernel, {input_, rois_});

      [computeEncoder setComputePipelineState:visionPSO];
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
      NSUInteger tgSize = visionPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > threadsPerBlock) {
        tgSize = threadsPerBlock;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(visionPSO);
    }
  });
  return output;
}

at::Tensor roi_align_backward_kernel(const at::Tensor& grad,
                                     const at::Tensor& rois,
                                     double spatial_scale,
                                     int64_t pooled_height,
                                     int64_t pooled_width,
                                     int64_t batch_size,
                                     int64_t channels,
                                     int64_t height,
                                     int64_t width,
                                     int64_t sampling_ratio,
                                     bool aligned) {
  using namespace at::native::mps;
  TORCH_CHECK(grad.is_mps(), "grad must be a MPS tensor");
  TORCH_CHECK(rois.is_mps(), "rois must be a MPS tensor");
  TORCH_CHECK(grad.scalar_type() != at::kHalf, "MPS does not support roi_align backward with float16 inputs.");

  at::TensorArg grad_t{grad, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  float spatial_scale_f = static_cast<float>(spatial_scale);

  at::Tensor grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  if (grad.numel() == 0) {
    return grad_input;
  }

  int64_t n_stride = grad.stride(0);
  int64_t c_stride = grad.stride(1);
  int64_t h_stride = grad.stride(2);
  int64_t w_stride = grad.stride(3);
  int64_t output_size = grad.numel();

  at::globalContext().alertNotDeterministic("roi_align_backward_kernel");
  auto rois_ = rois.contiguous();

  id<MTLBuffer> inputBuffer = getMTLBufferStorage(grad);
  id<MTLBuffer> roisBuffer = getMTLBufferStorage(rois_);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(grad_input);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize threadgroupsPerGrid = MTLSizeMake(
          std::min(ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)), static_cast<int64_t>(4096)),
          1,
          1);

      const std::string kernel = "roi_align_backward_" + scalarToMetalTypeString(grad.scalar_type());
      id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(visionPSO, kernel, {grad, rois_});

      [computeEncoder setComputePipelineState:visionPSO];
      // [N, C, H, W]
      [computeEncoder setBuffer:inputBuffer offset:grad.storage_offset() * grad.element_size() atIndex:0];
      [computeEncoder setBuffer:roisBuffer offset:rois_.storage_offset() * rois_.element_size() atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:grad_input.storage_offset() * grad_input.element_size() atIndex:2];

      [computeEncoder setBytes:&output_size length:sizeof(int64_t) atIndex:3];
      [computeEncoder setBytes:&channels length:sizeof(int64_t) atIndex:4];
      [computeEncoder setBytes:&height length:sizeof(int64_t) atIndex:5];
      [computeEncoder setBytes:&width length:sizeof(int64_t) atIndex:6];
      [computeEncoder setBytes:&pooled_height length:sizeof(int64_t) atIndex:7];
      [computeEncoder setBytes:&pooled_width length:sizeof(int64_t) atIndex:8];
      [computeEncoder setBytes:&sampling_ratio length:sizeof(int64_t) atIndex:9];
      [computeEncoder setBytes:&aligned length:sizeof(bool) atIndex:10];
      [computeEncoder setBytes:&spatial_scale_f length:sizeof(float) atIndex:11];
      [computeEncoder setBytes:&n_stride length:sizeof(int64_t) atIndex:12];
      [computeEncoder setBytes:&c_stride length:sizeof(int64_t) atIndex:13];
      [computeEncoder setBytes:&h_stride length:sizeof(int64_t) atIndex:14];
      [computeEncoder setBytes:&w_stride length:sizeof(int64_t) atIndex:15];

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
  return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::roi_align"), TORCH_FN(roi_align_forward_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchvision::_roi_align_backward"), TORCH_FN(roi_align_backward_kernel));
}

} // namespace ops
} // namespace vision
