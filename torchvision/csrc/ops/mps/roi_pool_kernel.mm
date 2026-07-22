#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include "mps_helpers.h"
#include "mps_kernels.h"

namespace vision {
namespace ops {

namespace {

std::tuple<at::Tensor, at::Tensor> roi_pool_forward_kernel(const at::Tensor& input,
                                                           const at::Tensor& rois,
                                                           double spatial_scale,
                                                           int64_t pooled_height,
                                                           int64_t pooled_width) {
  using namespace at::native::mps;
  TORCH_CHECK(input.is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(rois.is_mps(), "rois must be a MPS tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  int64_t num_rois = rois.size(0);
  int64_t channels = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  float spatial_scale_f = static_cast<float>(spatial_scale);

  at::Tensor output = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options().dtype(at::kLong));

  int64_t output_size = num_rois * pooled_height * pooled_width * channels;

  if (output.numel() == 0) {
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const std::string kernel = "roi_pool_" + scalarToMetalTypeString(input.scalar_type());
      id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(visionPSO, kernel, {input_, rois_});

      [computeEncoder setComputePipelineState:visionPSO];
      // [N, C, H, W]
      mtl_setArgs(computeEncoder,
                  input_,
                  rois_,
                  output,
                  argmax,
                  channels,
                  height,
                  width,
                  pooled_height,
                  pooled_width,
                  spatial_scale_f);

      MTLSize threadsPerGrid = MTLSizeMake(output_size, 1, 1);
      NSUInteger tgSize = std::min(static_cast<int64_t>(visionPSO.maxTotalThreadsPerThreadgroup), output_size);
      MTLSize threadGroupSize = MTLSizeMake(std::max<NSUInteger>(tgSize, 1), 1, 1);
      [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(visionPSO);
    }
  });
  return std::make_tuple(output, argmax);
}

at::Tensor roi_pool_backward_kernel(const at::Tensor& grad,
                                    const at::Tensor& rois,
                                    const at::Tensor& argmax,
                                    double spatial_scale,
                                    int64_t pooled_height,
                                    int64_t pooled_width,
                                    int64_t batch_size,
                                    int64_t channels,
                                    int64_t height,
                                    int64_t width) {
  using namespace at::native::mps;
  TORCH_CHECK(grad.is_mps(), "grad must be a MPS tensor");
  TORCH_CHECK(rois.is_mps(), "rois must be a MPS tensor");
  TORCH_CHECK(grad.scalar_type() != at::kHalf, "MPS does not support roi_pool backward with float16 inputs.");
  TORCH_CHECK(argmax.is_mps(), "argmax must be a MPS tensor");

  at::TensorArg grad_t{grad, "input", 1}, rois_t{rois, "rois", 2}, argmax_t{argmax, "argmax", 3};

  at::CheckedFrom c = "roi_pool_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t, argmax_t});
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

  at::globalContext().alertNotDeterministic("roi_pool_backward_kernel");
  auto argmax_ = argmax.contiguous(), rois_ = rois.contiguous();

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const std::string kernel = "roi_pool_backward_" + scalarToMetalTypeString(grad.scalar_type());
      id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(visionPSO, kernel, {grad, rois_, argmax_});

      [computeEncoder setComputePipelineState:visionPSO];
      // [N, C, H, W]
      mtl_setArgs(computeEncoder,
                  grad,
                  rois_,
                  argmax_,
                  grad_input,
                  channels,
                  height,
                  width,
                  pooled_height,
                  pooled_width,
                  spatial_scale_f,
                  n_stride,
                  c_stride,
                  h_stride,
                  w_stride);

      MTLSize threadsPerGrid = MTLSizeMake(output_size, 1, 1);
      NSUInteger tgSize = std::min(static_cast<int64_t>(visionPSO.maxTotalThreadsPerThreadgroup), output_size);
      MTLSize threadGroupSize = MTLSizeMake(std::max<NSUInteger>(tgSize, 1), 1, 1);
      [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(visionPSO);
    }
  });
  return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::roi_pool"), TORCH_FN(roi_pool_forward_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchvision::_roi_pool_backward"), TORCH_FN(roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision
