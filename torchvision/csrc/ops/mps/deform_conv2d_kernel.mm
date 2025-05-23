#include <ATen/ATen.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include "mps_kernels.h"

namespace vision {
namespace ops {

namespace {

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
  using namespace at::native::mps;
  at::Tensor input_c = input.contiguous();
  at::Tensor weight_c = weight.contiguous();
  at::Tensor offset_c = offset.contiguous();
  at::Tensor mask_c = mask.contiguous();
  at::Tensor bias_c = bias.contiguous();

  TORCH_CHECK(input_c.ndimension() == 4, "Input tensor must be 4D");
  TORCH_CHECK(weight_c.ndimension() == 4, "Weight tensor must be 4D");
  TORCH_CHECK(offset_c.ndimension() == 4, "Offset tensor must be 4D");
  TORCH_CHECK(!use_mask || mask_c.ndimension() == 4, "Mask tensor must be 4D if use_mask is true");
  TORCH_CHECK(input_c.is_mps(), "input must be a MPS tensor");

  at::DeviceGuard guard(input_c.device());

  int batch = input_c.size(0);
  int in_channels = input_c.size(1);
  int in_h = input_c.size(2);
  int in_w = input_c.size(3);
  int weight_h = weight_c.size(2);
  int weight_w = weight_c.size(3);
  int out_channels = weight_c.size(0);
  int ker_h = dilation_h * (weight_h - 1) + 1;
  int ker_w = dilation_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  TORCH_CHECK(weight_c.size(1) * n_weight_grps == in_channels,
    "Input channels (", in_channels, 
    ") must equal weight.size(1) * n_weight_grps (", weight_c.size(1), " * ", n_weight_grps, ")");
  TORCH_CHECK(weight_c.size(0) % n_weight_grps == 0,
    "Weight tensor's out channels (", weight_c.size(0), 
    ") must be divisible by n_weight_grps (", n_weight_grps, ")");
  TORCH_CHECK(offset_c.size(1) == n_offset_grps * 2 * weight_h * weight_w,
    "Offset tensor shape[1] is invalid: got ", offset_c.size(1), 
    ", expected ", n_offset_grps * 2 * weight_h * weight_w);
  TORCH_CHECK(!use_mask || mask_c.size(1) == n_offset_grps * weight_h * weight_w,
    "Mask tensor shape[1] is invalid: got ", mask_c.size(1), 
    ", expected ", n_offset_grps * weight_h * weight_w);
  TORCH_CHECK(in_channels % n_offset_grps == 0,
    "Input tensor channels (", in_channels, 
    ") must be divisible by n_offset_grps (", n_offset_grps, ")");
  TORCH_CHECK(offset_c.size(0) == batch,
    "Offset tensor batch size (", offset_c.size(0),
    ") must match input tensor batch size (", batch, ")");
  TORCH_CHECK(offset_c.size(2) == out_h && offset_c.size(3) == out_w,
    "Offset tensor spatial dimensions (", offset_c.size(2), ", ", offset_c.size(3), 
    ") must match calculated output dimensions (", out_h, ", ", out_w, ")");
  TORCH_CHECK(!use_mask || mask_c.size(0) == batch,
    "Mask tensor batch size (", mask_c.size(0),
    ") must match input tensor batch size (", batch, ")");
  TORCH_CHECK(!use_mask || (mask_c.size(2) == out_h && mask_c.size(3) == out_w),
    "Mask tensor spatial dimensions (", mask_c.size(2), ", ", mask_c.size(3),
    ") must match calculated output dimensions (", out_h, ", ", out_w, ")");
  TORCH_CHECK(out_h > 0 && out_w > 0,
    "Calculated output size too small - out_h: ", out_h, " out_w: ", out_w);

  auto columns = at::empty({in_channels * weight_h * weight_w, batch * out_h * out_w}, input_c.options());

  id<MTLBuffer> inputBuffer  = getMTLBufferStorage(input_c);
  id<MTLBuffer> offsetBuffer = getMTLBufferStorage(offset_c);
  id<MTLBuffer> maskBuffer   = use_mask ? getMTLBufferStorage(mask_c) : nil;
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(columns);

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  std::string kernelName = "deformable_im2col_" + scalarToMetalTypeString(input.scalar_type());
  id<MTLComputePipelineState> pipelineState = mps::visionPipelineState(device, kernelName);

  int num_kernels = in_channels * out_h * out_w * batch;
  NSUInteger threadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup;
  NSUInteger threadgroups = (num_kernels + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
  MTLSize threadGroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
  MTLSize threadgroupsPerGrid = MTLSizeMake(threadgroups, 1, 1);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^{
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      [computeEncoder setComputePipelineState:pipelineState];
      at::native::mps::mtl_setArgs(computeEncoder, inputBuffer, offsetBuffer, maskBuffer,
                                   in_h, in_w, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w, 
                                   dilation_h, dilation_w, batch, in_channels, n_offset_grps, out_h, out_w,
                                   use_mask, outputBuffer);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];
    }
  });
  int in_channels_per_grp = in_channels / n_weight_grps;
  int out_channels_per_grp = out_channels / n_weight_grps;
  auto weight_grouped = weight_c.view({n_weight_grps, out_channels_per_grp, in_channels_per_grp, weight_h, weight_w});
  auto columns_grouped = columns.view({n_weight_grps,
                                      (in_channels * weight_h * weight_w) / n_weight_grps,
                                      batch * out_h * out_w});
  auto weight_reshaped = weight_grouped.reshape({n_weight_grps, out_channels_per_grp, -1});
  auto out_grouped = at::bmm(weight_reshaped, columns_grouped);
  auto out = out_grouped.reshape({n_weight_grps * out_channels_per_grp, batch, out_h, out_w})
              .transpose(0, 1);
  return out + bias_c.view({1, out_channels, 1, 1});
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
      TORCH_FN(deform_conv2d_forward_kernel));
}

} // namespace ops
} // namespace vision