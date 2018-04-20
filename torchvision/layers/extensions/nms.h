#pragma once
#include "cpu/vision.h"

at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {

  auto result = dets.type().tensor();

  if (dets.type().is_cuda())
    std::runtime_error("NMS not implemented on the GPU");

  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
}
*/
