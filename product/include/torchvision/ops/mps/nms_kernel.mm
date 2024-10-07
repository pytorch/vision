#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include "mps_kernels.h"

namespace vision {
namespace ops {

namespace {

// This should be in sync with `nmsThreadsPerBlock` in the metal kernel.
constexpr int64_t nmsThreadsPerBlock = sizeof(uint64_t) * 8;

at::Tensor nms_kernel(const at::Tensor& dets, const at::Tensor& scores, double iou_threshold) {
  using namespace at::native::mps;
  TORCH_CHECK(dets.is_mps(), "dets must be a MPS tensor");
  TORCH_CHECK(scores.is_mps(), "scores must be a MPS tensor");

  TORCH_CHECK(dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(dets.size(1) == 4, "boxes should have 4 elements in dimension 1, got ", dets.size(1));
  TORCH_CHECK(scores.dim() == 1, "scores should be a 1d tensor, got ", scores.dim(), "D");
  TORCH_CHECK(dets.size(0) == scores.size(0),
              "boxes and scores should have same number of elements in ",
              "dimension 0, got ",
              dets.size(0),
              " and ",
              scores.size(0))

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t).contiguous();
  int64_t dets_num = dets.size(0);
  float iou_threshold_f = static_cast<float>(iou_threshold);

  const int col_blocks = (dets_num + nmsThreadsPerBlock - 1) / nmsThreadsPerBlock;
  at::Tensor mask = at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  id<MTLBuffer> inputBuffer = getMTLBufferStorage(dets_sorted);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(mask);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize threadgroupsPerGrid = MTLSizeMake(col_blocks, col_blocks, 1);

      const std::string kernel = "nms_" + scalarToMetalTypeString(dets_sorted.scalar_type());
      id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(visionPSO, kernel, {dets, scores});

      [computeEncoder setComputePipelineState:visionPSO];
      [computeEncoder setBuffer:inputBuffer offset:dets_sorted.storage_offset() * dets_sorted.element_size() atIndex:0];
      [computeEncoder setBuffer:outputBuffer offset:mask.storage_offset() * mask.element_size() atIndex:1];
      [computeEncoder setBytes:&dets_num length:sizeof(int64_t) atIndex:2];
      [computeEncoder setBytes:&iou_threshold_f length:sizeof(float) atIndex:3];

      // A threadGroup is equivalent to a cuda's block.
      NSUInteger tgSize = visionPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > nmsThreadsPerBlock) {
        tgSize = nmsThreadsPerBlock;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(visionPSO);
    }
  });

  int64_t num_to_keep = 0;

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  for (int64_t i = 0; i < dets_num; i++) {
    int64_t nblock = i / nmsThreadsPerBlock;
    int64_t inblock = i % nmsThreadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int64_t j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(order_t.device(), keep.scalar_type())});
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
}

} // namespace ops
} // namespace vision
