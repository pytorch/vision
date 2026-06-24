#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../StableABICompat.h"
#include "nms_metal_shader.h"

namespace vision {
namespace ops {

namespace {

using torch::stable::Tensor;

// This should be in sync with `nmsThreadsPerBlock` in the metal kernel.
constexpr int64_t nmsThreadsPerBlock = sizeof(uint64_t) * 8;

// Lazily compile the nms Metal shader library once for the process, mirroring
// the lazy-singleton the AOTInductor MPS backend generates. The handle lives
// for the process lifetime (as the legacy static MetalShaderLibrary did).
AOTIMetalShaderLibraryHandle nms_shader_library() {
  static AOTIMetalShaderLibraryHandle library = []() {
    AOTIMetalShaderLibraryHandle handle = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_mps_create_shader_library(nms_metal_shader, &handle));
    return handle;
  }();
  return library;
}

// Mirrors at::native::mps::scalarToMetalTypeString for the dtypes nms registers
// (nms_float / nms_half). An unsupported dtype yields a name with no matching
// kernel, so the lookup in aoti_torch_mps_get_kernel_function fails -- as the
// legacy visionPipelineState lookup did for unsupported types.
const char* metal_type_string(torch::headeronly::ScalarType scalar_type) {
  if (scalar_type == torch::headeronly::ScalarType::Float) {
    return "float";
  }
  if (scalar_type == torch::headeronly::ScalarType::Half) {
    return "half";
  }
  return "";
}

// Arguments bound to the nms kernel inside the command block. The tensors are
// owned by the caller (nms_kernel) and outlive the synchronous dispatch.
struct NmsLaunchArgs {
  AtenTensorHandle dets_sorted;
  AtenTensorHandle mask;
  AtenTensorHandle iou_threshold;
  int64_t dets_num;
  uint64_t grid[2];
  uint64_t threadgroup[2];
};

// Trampoline invoked on the MPS stream's queue by aoti_torch_mps_run_command_block.
void nms_encode(AOTIMetalKernelFunctionHandle func, void* user_data) {
  const auto* launch_args = static_cast<const NmsLaunchArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 0, launch_args->dets_sorted));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 1, launch_args->mask));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_int(func, 2, launch_args->dets_num));
  // The MPS shim has no scalar-float arg setter, so iou_threshold rides in as a
  // 1-element float32 tensor bound at buffer(3).
  // TODO(stable-abi): bind a float directly once aoti_torch_mps_set_arg_double /
  // set_arg_bytes lands upstream (pytorch/pytorch).
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_set_arg_tensor(func, 3, launch_args->iou_threshold));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_array_with_group_size(
      func,
      launch_args->grid,
      /*length_size=*/2,
      launch_args->threadgroup,
      /*group_size_size=*/2));
}

Tensor nms_kernel(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold) {
  STD_TORCH_CHECK(
      dets.device().type() == torch::headeronly::DeviceType::MPS,
      "dets must be a MPS tensor");
  STD_TORCH_CHECK(
      scores.device().type() == torch::headeronly::DeviceType::MPS,
      "scores must be a MPS tensor");

  STD_TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  STD_TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  STD_TORCH_CHECK(
      scores.dim() == 1, "scores should be a 1d tensor, got ", scores.dim(), "D");
  STD_TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  if (dets.numel() == 0) {
    return torch::stable::new_empty(dets, {0}, torch::headeronly::ScalarType::Long);
  }

  auto order_t = std::get<1>(
      stable_helpers::sort(scores, /*stable=*/true, /*dim=*/0, /*descending=*/true));
  auto dets_sorted =
      torch::stable::contiguous(stable_helpers::index_select(dets, 0, order_t));
  int64_t dets_num = dets.size(0);

  const int64_t col_blocks =
      (dets_num + nmsThreadsPerBlock - 1) / nmsThreadsPerBlock;
  Tensor mask = torch::stable::new_empty(
      dets, {dets_num * col_blocks}, torch::headeronly::ScalarType::Long);

  // The MPS kernel reads iou_threshold as a float; carry it in a 1-element
  // float32 tensor (see note in nms_encode).
  Tensor iou_threshold_t =
      torch::stable::new_empty(dets, {1}, torch::headeronly::ScalarType::Float);
  torch::stable::fill_(iou_threshold_t, iou_threshold);

  const std::string kernel =
      "nms_" + std::string(metal_type_string(dets_sorted.scalar_type()));
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(
      nms_shader_library(), kernel.c_str(), &func));

  // A threadGroup is equivalent to a cuda's block; dispatch col_blocks x
  // col_blocks threadgroups of nmsThreadsPerBlock threads. The shim dispatches
  // by total threads, so the x grid extent is scaled by the threadgroup width.
  NmsLaunchArgs launch_args{
      dets_sorted.get(),
      mask.get(),
      iou_threshold_t.get(),
      dets_num,
      {static_cast<uint64_t>(col_blocks) * nmsThreadsPerBlock,
       static_cast<uint64_t>(col_blocks)},
      {static_cast<uint64_t>(nmsThreadsPerBlock), 1}};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_run_command_block(func, &nms_encode, &launch_args));

  int64_t num_to_keep = 0;

  Tensor mask_cpu = torch::stable::to(
      mask, torch::stable::Device(torch::headeronly::DeviceType::CPU));
  unsigned long long* mask_host =
      (unsigned long long*)mask_cpu.const_data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  Tensor keep = torch::stable::new_empty(
      dets,
      {dets_num},
      torch::headeronly::ScalarType::Long,
      /*layout=*/std::nullopt,
      torch::stable::Device(torch::headeronly::DeviceType::CPU));
  int64_t* keep_out = keep.mutable_data_ptr<int64_t>();

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

  return stable_helpers::index_select(
      order_t,
      0,
      torch::stable::to(
          torch::stable::narrow(keep, /*dim=*/0, /*start=*/0, /*length=*/num_to_keep),
          order_t.device()));
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl("nms", TORCH_BOX(&nms_kernel));
}

} // namespace ops
} // namespace vision
