#include <ATen/core/op_registration/op_registration.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "DeformConv.h"
#include "PSROIAlign.h"
#include "PSROIPool.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "empty_tensor_op.h"
#include "nms.h"

C10_EXPORT int64_t _cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

namespace vision {
int RegisterOps() noexcept {
  static auto registry =
      torch::RegisterOperators()
          .op("torchvision::nms", &nms)
          .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> Tensor",
              &roi_align)
          .op("torchvision::roi_pool", &roi_pool)
          .op("torchvision::_new_empty_tensor_op", &new_empty_tensor)
          .op("torchvision::ps_roi_align", &ps_roi_align)
          .op("torchvision::ps_roi_pool", &ps_roi_pool)
          .op("torchvision::deform_conv2d", &deform_conv2d)
          .op("torchvision::_cuda_version", &_cuda_version);
  return 0;
}
} // namespace vision
