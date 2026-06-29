#include "encode_jpegs_cuda.h"

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/util/Exception.h>

#if !NVJPEG_FOUND
namespace vision {
namespace image {
std::vector<torch::stable::Tensor> encode_jpegs_cuda(
    const std::vector<torch::stable::Tensor>& decoded_images,
    const int64_t quality) {
  STD_TORCH_CHECK(
      false, "encode_jpegs_cuda: torchvision not compiled with nvJPEG support");
}
} // namespace image
} // namespace vision

#else
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <memory>
#include <mutex>
#include <optional>
namespace vision {
namespace image {

// We use a global to cache the encoder instance and reuse it across calls to
// the corresponding pytorch function.
std::mutex encoderMutex;
std::unique_ptr<CUDAJpegEncoder> cudaJpegEncoder;

// Make waitingStream wait until all work currently enqueued on runningStream
// has completed.
// https://github.com/meta-pytorch/torchcodec/blob/1dc85b7a7900d91fee207ccdc02f211a051688fe/src/torchcodec/_core/CUDACommon.cpp#L30-L47
static void syncStreams(
    cudaStream_t runningStream,
    cudaStream_t waitingStream) {
  cudaEvent_t event;
  cudaError_t err = cudaEventCreate(&event);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));
  err = cudaEventRecord(event, runningStream);
  STD_TORCH_CHECK(
      err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));
  err = cudaStreamWaitEvent(waitingStream, event, 0);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "cudaStreamWaitEvent failed: ",
      cudaGetErrorString(err));
  cudaEventDestroy(event);
}

std::vector<torch::stable::Tensor> encode_jpegs_cuda(
    const std::vector<torch::stable::Tensor>& decoded_images,
    const int64_t quality) {
  // Some nvjpeg structures are not thread safe so we're keeping it single
  // threaded for now. In the future this may be an opportunity to unlock
  // further speedups
  std::lock_guard<std::mutex> lock(encoderMutex);
  STD_TORCH_CHECK(decoded_images.size() > 0, "Empty input tensor list");
  torch::stable::Device device = decoded_images[0].device();
  STD_TORCH_CHECK(
      device.is_cuda(),
      "All input tensors must be on a CUDA device when encoding with nvjpeg");
  torch::stable::accelerator::DeviceGuard device_guard(device.index());

  // lazy init of the encoder class
  // the encoder object holds on to a lot of state and is expensive to create,
  // so we reuse it across calls. NB: the cached structures are device specific
  // and cannot be reused across devices
  if (cudaJpegEncoder == nullptr || device != cudaJpegEncoder->target_device) {
    if (cudaJpegEncoder != nullptr) {
      cudaJpegEncoder.reset(new CUDAJpegEncoder(device));
    } else {
      cudaJpegEncoder = std::make_unique<CUDAJpegEncoder>(device);

      // Unfortunately, we cannot rely on the smart pointer releasing the
      // encoder object correctly upon program exit. This is because, when
      // cudaJpegEncoder gets destroyed, the CUDA runtime may already be shut
      // down, rendering all destroy* calls in the encoder destructor invalid.
      // Instead, we use an atexit hook which executes after main() finishes,
      // but hopefully before CUDA shuts down when the program exits. If CUDA is
      // already shut down the destructor will detect this and will not attempt
      // to destroy any encoder structures.
      std::atexit([]() { cudaJpegEncoder.reset(); });
    }
  }

  std::vector<torch::stable::Tensor> contig_images;
  contig_images.reserve(decoded_images.size());
  for (const auto& image : decoded_images) {
    STD_TORCH_CHECK(
        image.scalar_type() == torch::headeronly::ScalarType::Byte,
        "Input tensor dtype should be uint8");

    STD_TORCH_CHECK(
        image.device() == device,
        "All input tensors must be on the same CUDA device when encoding with nvjpeg")

    STD_TORCH_CHECK(
        image.dim() == 3 && image.numel() > 0,
        "Input data should be a 3-dimensional tensor");

    STD_TORCH_CHECK(
        image.size(0) == 3,
        "The number of channels should be 3, got: ",
        image.size(0));

    // nvjpeg requires images to be contiguous
    contig_images.push_back(torch::stable::contiguous(image));
  }

  cudaJpegEncoder->set_quality(quality);
  std::vector<torch::stable::Tensor> encoded_images;
  for (const auto& image : contig_images) {
    auto encoded_image = cudaJpegEncoder->encode_jpeg(image);
    encoded_images.push_back(encoded_image);
  }

  // We use a dedicated stream to do the encoding and even though the results
  // may be ready on that stream we cannot assume that they are also available
  // on the current stream of the calling context when this function returns. We
  // use a blocking event to ensure that this is indeed the case. Crucially, we
  // do not want to block the host at this particular point
  // (which is what cudaStreamSynchronize would do.) Events allow us to
  // synchronize the streams without blocking the host.
  syncStreams(cudaJpegEncoder->stream, cudaJpegEncoder->current_stream);
  return encoded_images;
}

CUDAJpegEncoder::CUDAJpegEncoder(const torch::stable::Device& target_device)
    : original_device(
          torch::headeronly::DeviceType::CUDA,
          torch::stable::accelerator::getCurrentDeviceIndex()),
      target_device(target_device) {
  torch::stable::accelerator::DeviceGuard device_guard(target_device.index());
  // Pool-owned (not a raw leaked) stream; avoids a cross-DSO teardown hazard.
  // https://github.com/pytorch/pytorch/blob/98e36864e640023a716e058d894ea2d20e76e5f7/torch/csrc/stable/c/shim.h#L127-L130
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_get_cuda_stream_from_pool(
      false, target_device.index(), &stream_ptr));
  stream = static_cast<cudaStream_t>(stream_ptr);

  // The caller's current stream, captured at construction. encode_jpegs_cuda
  // makes it wait on our private `stream` (via syncStreams) before returning.
  // https://github.com/pytorch/pytorch/blob/98e36864e640023a716e058d894ea2d20e76e5f7/torch/csrc/inductor/aoti_torch/c/shim.h#L573-L602
  void* current_stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      original_device.index(), &current_stream_ptr));
  current_stream = static_cast<cudaStream_t>(current_stream_ptr);

  nvjpegStatus_t status;
  status = nvjpegCreateSimple(&nvjpeg_handle);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg handle: ",
      status);

  status = nvjpegEncoderStateCreate(nvjpeg_handle, &nv_enc_state, stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg encoder state: ",
      status);

  status = nvjpegEncoderParamsCreate(nvjpeg_handle, &nv_enc_params, stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg encoder params: ",
      status);
}

CUDAJpegEncoder::~CUDAJpegEncoder() {
  /*
  The below code works on Mac and Linux, but fails on Windows.
  This is because on Windows, the atexit hook which calls this
  destructor executes after cuda is already shut down causing SIGSEGV.
  We do not have a solution to this problem at the moment, so we'll
  just leak the libnvjpeg & cuda variables for the time being and hope
  that the CUDA runtime handles cleanup for us.
  Please send a PR if you have a solution for this problem.
  */

  // // We run cudaGetDeviceCount as a dummy to test if the CUDA runtime is
  // still
  // // initialized. If it is not, we can skip the rest of this function as it
  // is
  // // unsafe to execute.
  // int deviceCount = 0;
  // cudaError_t error = cudaGetDeviceCount(&deviceCount);
  // if (error != cudaSuccess)
  //   return; // CUDA runtime has already shut down. There's nothing we can do
  //           // now.

  // nvjpegStatus_t status;

  // status = nvjpegEncoderParamsDestroy(nv_enc_params);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg encoder params: ",
  //     status);

  // status = nvjpegEncoderStateDestroy(nv_enc_state);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg encoder state: ",
  //     status);

  // cudaStreamSynchronize(stream);

  // status = nvjpegDestroy(nvjpeg_handle);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS, "nvjpegDestroy failed: ", status);
}

torch::stable::Tensor CUDAJpegEncoder::encode_jpeg(
    const torch::stable::Tensor& src_image) {
  nvjpegStatus_t status;
  cudaError_t cudaStatus;

  // Ensure that the incoming src_image is safe to use
  cudaStatus = cudaStreamSynchronize(current_stream);
  STD_TORCH_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);

  int channels = src_image.size(0);
  int height = src_image.size(1);
  int width = src_image.size(2);

  status = nvjpegEncoderParamsSetSamplingFactors(
      nv_enc_params, NVJPEG_CSS_444, stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params sampling factors: ",
      status);

  nvjpegImage_t target_image;
  for (int c = 0; c < channels; c++) {
    target_image.channel[c] =
        torch::stable::select(src_image, 0, c).mutable_data_ptr<uint8_t>();
    // this is why we need contiguous tensors
    target_image.pitch[c] = width;
  }
  for (int c = channels; c < NVJPEG_MAX_COMPONENT; c++) {
    target_image.channel[c] = nullptr;
    target_image.pitch[c] = 0;
  }
  // Encode the image
  status = nvjpegEncodeImage(
      nvjpeg_handle,
      nv_enc_state,
      nv_enc_params,
      &target_image,
      NVJPEG_INPUT_RGB,
      width,
      height,
      stream);

  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS, "image encoding failed: ", status);
  // Retrieve length of the encoded image
  size_t length;
  status = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle, nv_enc_state, NULL, &length, stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image stream state: ",
      status);

  // Synchronize the stream to ensure that the encoded image is ready
  cudaStatus = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);

  // Reserve buffer for the encoded image
  torch::stable::Tensor encoded_image = torch::stable::empty(
      {static_cast<int64_t>(length)},
      torch::headeronly::ScalarType::Byte,
      std::nullopt,
      target_device);
  cudaStatus = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(cudaStatus == cudaSuccess, "CUDA ERROR: ", cudaStatus);
  // Retrieve the encoded image
  status = nvjpegEncodeRetrieveBitstreamDevice(
      nvjpeg_handle,
      nv_enc_state,
      encoded_image.mutable_data_ptr<uint8_t>(),
      &length,
      stream);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to retrieve encoded image: ",
      status);
  return encoded_image;
}

void CUDAJpegEncoder::set_quality(const int64_t quality) {
  nvjpegStatus_t paramsQualityStatus =
      nvjpegEncoderParamsSetQuality(nv_enc_params, quality, stream);
  STD_TORCH_CHECK(
      paramsQualityStatus == NVJPEG_STATUS_SUCCESS,
      "Failed to set nvjpeg encoder params quality: ",
      paramsQualityStatus);
}

} // namespace image
} // namespace vision

#endif // NVJPEG_FOUND

namespace vision {
namespace image {

STABLE_TORCH_LIBRARY_FRAGMENT(image, m) {
  m.def("encode_jpegs_cuda(Tensor[] decoded_images, int quality) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(image, CompositeExplicitAutograd, m) {
  m.impl("encode_jpegs_cuda", TORCH_BOX(&encode_jpegs_cuda));
}

} // namespace image
} // namespace vision
