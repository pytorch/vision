#include "encode_decode_jpegs_cuda.h"

#if NVJPEG_FOUND

namespace vision {
namespace image {
nvjpegHandle_t nvjpeg_handle = nullptr;
std::once_flag nvjpeg_handle_creation_flag;

void nvjpeg_init() {
  if (nvjpeg_handle == nullptr) {
    nvjpegStatus_t create_status = nvjpegCreateSimple(&nvjpeg_handle);

    if (create_status != NVJPEG_STATUS_SUCCESS) {
      // Reset handle so that one can still call the function again in the
      // same process if there was a failure
      free(nvjpeg_handle);
      nvjpeg_handle = nullptr;
    }
    TORCH_CHECK(
        create_status == NVJPEG_STATUS_SUCCESS,
        "nvjpegCreateSimple failed: ",
        create_status);
  }
}
} // namespace image
} // namespace vision
#endif
