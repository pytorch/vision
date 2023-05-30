//#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include "vision_kernels.h"

namespace vision {
namespace ops {

namespace {

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {

  using namespace at::native::mps;
  TORCH_CHECK(dets.is_mps(), "dets must be a MPS tensor");
  TORCH_CHECK(scores.is_mps(), "scores must be a MPS tensor");

  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0))
  
  //at::Tensor input = at::arange({10}, at::kFloat, c10::nullopt, at::kMPS, c10::nullopt);
  //at::Tensor other = at::arange({10}, at::kFloat, c10::nullopt, at::kMPS, c10::nullopt);
  //at::Tensor out = at::zeros({10}, at::kFloat, c10::nullopt, at::kMPS, c10::nullopt);
  
  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t).contiguous();
  int dets_num = dets.size(0);
  float iou_threshold_f = static_cast<float>(iou_threshold);

  //TODO: ceil_div
  //const int col_blocks = ceil_div(dets_num, threadsPerBlock);
  //at::Tensor mask =
  //  at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));
  at::Tensor mask =
    at::empty({dets_num}, dets.options().dtype(at::kLong));

  id<MTLBuffer> inputBuffer = getMTLBufferStorage(dets_sorted);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(mask);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  //const uint32_t nDim = iter.ndim();
  //constexpr uint32_t nOffsets = 3;
  const uint32_t numThreads = dets_num;
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      NSError* error = nil;
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);


      const std::string kernel = "nms_" + scalarToMetalTypeString(dets_sorted.scalar_type());
      id<MTLComputePipelineState> binaryPSO = mps::binaryPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      //getMPSProfiler().beginProfileKernel(binaryPSO, kernel, {input, other});

      [computeEncoder setComputePipelineState:binaryPSO];
      [computeEncoder setBuffer:inputBuffer offset:dets_sorted.storage_offset() * dets_sorted.element_size() atIndex:0];
      [computeEncoder setBuffer:outputBuffer offset:mask.storage_offset() * mask.element_size() atIndex:1];
      [computeEncoder setBytes:&dets_num length:sizeof(int) atIndex:2];
      [computeEncoder setBytes:&iou_threshold_f length:sizeof(float) atIndex:3];

      NSUInteger tgSize = binaryPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
        tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

      //getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
  return mask;

}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
}

} // namespace ops
} // namespace vision
