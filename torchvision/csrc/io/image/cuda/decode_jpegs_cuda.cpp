#include "decode_jpegs_cuda.h"
#if !NVJPEG_FOUND && !ROCJPEG_FOUND
namespace vision {
namespace image {
std::vector<torch::Tensor> decode_jpegs_cuda(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode,
    torch::Device device) {
  STD_TORCH_CHECK(
      false, "decode_jpegs_cuda: torchvision not compiled with nvJPEG support");
}
} // namespace image
} // namespace vision
#endif

#if NVJPEG_FOUND || ROCJPEG_FOUND
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace vision {
namespace image {
namespace {
std::mutex decoderMutex;
std::unique_ptr<GpuJpegDecoder> gpuJpegDecoder;

std::vector<torch::Tensor> validate_and_make_contiguous(
    const std::vector<torch::Tensor>& encoded_images) {
  std::vector<torch::Tensor> contig_images;
  contig_images.reserve(encoded_images.size());
  for (auto& encoded_image : encoded_images) {
    STD_TORCH_CHECK(
        encoded_image.dtype() == torch::kU8, "Expected a torch.uint8 tensor");
    STD_TORCH_CHECK(
        !encoded_image.is_cuda(), "The input tensor must be on CPU");
    STD_TORCH_CHECK(
        encoded_image.dim() == 1 && encoded_image.numel() > 0,
        "Expected a non empty 1-dimensional tensor");
    // The decoder backends require contiguous input; contiguous() is a no-op
    // when the tensor already is.
    contig_images.push_back(encoded_image.contiguous());
  }
  return contig_images;
}
} // namespace

std::vector<torch::Tensor> decode_jpegs_cuda(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode,
    torch::Device device) {
  C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cuda.decode_jpegs_cuda.decode_jpegs_cuda");

  std::lock_guard<std::mutex> lock(decoderMutex);

  STD_TORCH_CHECK(
      device.is_cuda(), "Expected the device parameter to be a cuda device");

  std::vector<torch::Tensor> contig_images =
      validate_and_make_contiguous(encoded_images);

  at::cuda::CUDAGuard device_guard(device);

  if (gpuJpegDecoder == nullptr || device != gpuJpegDecoder->target_device) {
    if (gpuJpegDecoder != nullptr) {
      gpuJpegDecoder.reset(new GpuJpegDecoder(device));
    } else {
      gpuJpegDecoder = std::make_unique<GpuJpegDecoder>(device);
      std::atexit([]() { gpuJpegDecoder.reset(); });
    }
  }

  try {
    return gpuJpegDecoder->decode_images(contig_images, mode);
  } catch (const std::exception& e) {
    STD_TORCH_CHECK(false, "Error while decoding JPEG images: ", e.what());
  }
}

} // namespace image
} // namespace vision
#endif

#if NVJPEG_FOUND
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <memory>
#include <string>
#include <typeinfo>
namespace vision {
namespace image {

std::vector<torch::Tensor> CUDAJpegDecoder::decode_images(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode) {
  int major_version;
  int minor_version;
  nvjpegStatus_t get_major_property_status =
      nvjpegGetProperty(MAJOR_VERSION, &major_version);
  nvjpegStatus_t get_minor_property_status =
      nvjpegGetProperty(MINOR_VERSION, &minor_version);

  STD_TORCH_CHECK(
      get_major_property_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegGetProperty failed: ",
      get_major_property_status);
  STD_TORCH_CHECK(
      get_minor_property_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegGetProperty failed: ",
      get_minor_property_status);
  if ((major_version < 11) || ((major_version == 11) && (minor_version < 6))) {
    TORCH_WARN_ONCE(
        "There is a memory leak issue in the nvjpeg library for CUDA versions < 11.6. "
        "Make sure to rely on CUDA 11.6 or above before using decode_jpeg(..., device='cuda').");
  }

  nvjpegOutputFormat_t output_format;

  switch (mode) {
    case vision::image::IMAGE_READ_MODE_UNCHANGED:
      // Using NVJPEG_OUTPUT_UNCHANGED causes differently sized output channels
      // which is related to the subsampling used I'm not sure why this is the
      // case, but for now we're just using RGB and later removing channels from
      // grayscale images.
      output_format = NVJPEG_OUTPUT_UNCHANGED;
      break;
    case vision::image::IMAGE_READ_MODE_GRAY:
      output_format = NVJPEG_OUTPUT_Y;
      break;
    case vision::image::IMAGE_READ_MODE_RGB:
      output_format = NVJPEG_OUTPUT_RGB;
      break;
    default:
      STD_TORCH_CHECK(
          false, "The provided mode is not supported for JPEG decoding on GPU");
  }

  at::cuda::CUDAEvent event;
  auto result = decode_images(encoded_images, output_format);
  auto current_stream{
      target_device.has_index()
          ? at::cuda::getCurrentCUDAStream(original_device.index())
          : at::cuda::getCurrentCUDAStream()};
  event.record(stream);
  event.block(current_stream);
  return result;
}

CUDAJpegDecoder::CUDAJpegDecoder(const torch::Device& target_device)
    : original_device{torch::kCUDA, c10::cuda::current_device()},
      target_device{target_device},
      stream{
          target_device.has_index()
              ? at::cuda::getStreamFromPool(false, target_device.index())
              : at::cuda::getStreamFromPool(false)} {
  nvjpegStatus_t status;

  hw_decode_available = true;
  status = nvjpegCreateEx(
      NVJPEG_BACKEND_HARDWARE,
      NULL,
      NULL,
      NVJPEG_FLAGS_DEFAULT,
      &nvjpeg_handle);
  if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
    status = nvjpegCreateEx(
        NVJPEG_BACKEND_DEFAULT,
        NULL,
        NULL,
        NVJPEG_FLAGS_DEFAULT,
        &nvjpeg_handle);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with default backend: ",
        status);
    hw_decode_available = false;
  } else {
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with hardware backend: ",
        status);
  }

  status = nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg state: ",
      status);

  status = nvjpegDecoderCreate(
      nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder: ",
      status);

  status = nvjpegDecoderStateCreate(
      nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder state: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[0]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[1]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferDeviceCreate(nvjpeg_handle, NULL, &device_buffer);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create device buffer: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[0]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[1]);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create decode params: ",
      status);
}

CUDAJpegDecoder::~CUDAJpegDecoder() {
  /*
  The below code works on Mac and Linux, but fails on Windows.
  This is because on Windows, the atexit hook which calls this
  destructor executes after cuda is already shut down causing SIGSEGV.
  We do not have a solution to this problem at the moment, so we'll
  just leak the libnvjpeg & cuda variables for the time being and hope
  that the CUDA runtime handles cleanup for us.
  Please send a PR if you have a solution for this problem.
  */

  // nvjpegStatus_t status;

  // status = nvjpegDecodeParamsDestroy(nvjpeg_decode_params);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg decode params: ",
  //     status);

  // status = nvjpegJpegStreamDestroy(jpeg_streams[0]);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy jpeg stream: ",
  //     status);

  // status = nvjpegJpegStreamDestroy(jpeg_streams[1]);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy jpeg stream: ",
  //     status);

  // status = nvjpegBufferPinnedDestroy(pinned_buffers[0]);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy pinned buffer[0]: ",
  //     status);

  // status = nvjpegBufferPinnedDestroy(pinned_buffers[1]);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy pinned buffer[1]: ",
  //     status);

  // status = nvjpegBufferDeviceDestroy(device_buffer);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy device buffer: ",
  //     status);

  // status = nvjpegJpegStateDestroy(nvjpeg_decoupled_state);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg decoupled state: ",
  //     status);

  // status = nvjpegDecoderDestroy(nvjpeg_decoder);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg decoder: ",
  //     status);

  // status = nvjpegJpegStateDestroy(nvjpeg_state);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg state: ",
  //     status);

  // status = nvjpegDestroy(nvjpeg_handle);
  // STD_TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS, "nvjpegDestroy failed: ", status);
}

std::tuple<
    std::vector<nvjpegImage_t>,
    std::vector<torch::Tensor>,
    std::vector<int>>
CUDAJpegDecoder::prepare_buffers(
    const std::vector<torch::Tensor>& encoded_images,
    const nvjpegOutputFormat_t& output_format) {
  /*
    This function scans the encoded images' jpeg headers and
    allocates decoding buffers based on the metadata found

    Args:
    - encoded_images (std::vector<torch::Tensor>): a vector of tensors
    containing the jpeg bitstreams to be decoded. Each tensor must have dtype
    torch.uint8 and device cpu
    - output_format (nvjpegOutputFormat_t): NVJPEG_OUTPUT_RGB, NVJPEG_OUTPUT_Y
    or NVJPEG_OUTPUT_UNCHANGED

    Returns:
    - decoded_images (std::vector<nvjpegImage_t>): a vector of nvjpegImages
    containing pointers to the memory of the decoded images
    - output_tensors (std::vector<torch::Tensor>): a vector of Tensors
    containing the decoded images. `decoded_images` points to the memory of
    output_tensors
    - channels (std::vector<int>): a vector of ints containing the number of
    output image channels for every image
  */

  int width[NVJPEG_MAX_COMPONENT];
  int height[NVJPEG_MAX_COMPONENT];
  std::vector<int> channels(encoded_images.size());
  nvjpegChromaSubsampling_t subsampling;
  nvjpegStatus_t status;

  std::vector<torch::Tensor> output_tensors{encoded_images.size()};
  std::vector<nvjpegImage_t> decoded_images{encoded_images.size()};

  for (std::vector<at::Tensor>::size_type i = 0; i < encoded_images.size();
       i++) {
    // extract bitstream meta data to figure out the number of channels, height,
    // width for every image
    status = nvjpegGetImageInfo(
        nvjpeg_handle,
        (unsigned char*)encoded_images[i].data_ptr(),
        encoded_images[i].numel(),
        &channels[i],
        &subsampling,
        width,
        height);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to get image info: ", status);

    STD_TORCH_CHECK(
        subsampling != NVJPEG_CSS_UNKNOWN, "Unknown chroma subsampling");

    // output channels may be different from the actual number of channels in
    // the image, e.g. we decode a grayscale image as RGB and slice off the
    // extra channels later
    int output_channels = 3;
    if (output_format == NVJPEG_OUTPUT_RGB ||
        output_format == NVJPEG_OUTPUT_UNCHANGED) {
      output_channels = 3;
    } else if (output_format == NVJPEG_OUTPUT_Y) {
      output_channels = 1;
    }

    // reserve output buffer
    auto output_tensor = torch::empty(
        {int64_t(output_channels), int64_t(height[0]), int64_t(width[0])},
        torch::dtype(torch::kU8).device(target_device));
    output_tensors[i] = output_tensor;

    // fill nvjpegImage_t struct
    for (int c = 0; c < output_channels; c++) {
      decoded_images[i].channel[c] = output_tensor[c].data_ptr<uint8_t>();
      decoded_images[i].pitch[c] = width[0];
    }
    for (int c = output_channels; c < NVJPEG_MAX_COMPONENT; c++) {
      decoded_images[i].channel[c] = NULL;
      decoded_images[i].pitch[c] = 0;
    }
  }
  return {decoded_images, output_tensors, channels};
}

std::vector<torch::Tensor> CUDAJpegDecoder::decode_images(
    const std::vector<torch::Tensor>& encoded_images,
    const nvjpegOutputFormat_t& output_format) {
  /*
    This function decodes a batch of jpeg bitstreams.
    We scan all encoded bitstreams and sort them into two groups:
    1. Baseline JPEGs: Can be decoded with hardware support on A100+ GPUs.
    2. Other JPEGs (e.g. progressive JPEGs): Can also be decoded on the
    GPU (albeit with software support only) but need some preprocessing on the
    host first.

    See
    https://github.com/NVIDIA/CUDALibrarySamples/blob/f17940ac4e705bf47a8c39f5365925c1665f6c98/nvJPEG/nvJPEG-Decoder/nvjpegDecoder.cpp#L33
    for reference.

    Args:
    - encoded_images (std::vector<torch::Tensor>): a vector of tensors
    containing the jpeg bitstreams to be decoded
    - output_format (nvjpegOutputFormat_t): NVJPEG_OUTPUT_RGB, NVJPEG_OUTPUT_Y
    or NVJPEG_OUTPUT_UNCHANGED
    - device (torch::Device): The desired CUDA device for the returned Tensors

    Returns:
    - output_tensors (std::vector<torch::Tensor>): a vector of Tensors
    containing the decoded images
  */

  auto [decoded_imgs_buf, output_tensors, channels] =
      prepare_buffers(encoded_images, output_format);

  nvjpegStatus_t status;
  cudaError_t cudaStatus;

  cudaStatus = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cudaStatus);

  // baseline JPEGs can be batch decoded with hardware support on A100+ GPUs
  // ultra fast!
  std::vector<const unsigned char*> hw_input_buffer;
  std::vector<size_t> hw_input_buffer_size;
  std::vector<nvjpegImage_t> hw_output_buffer;

  // other JPEG types such as progressive JPEGs can be decoded one-by-one in
  // software slow :(
  std::vector<const unsigned char*> sw_input_buffer;
  std::vector<size_t> sw_input_buffer_size;
  std::vector<nvjpegImage_t> sw_output_buffer;

  if (hw_decode_available) {
    for (std::vector<at::Tensor>::size_type i = 0; i < encoded_images.size();
         ++i) {
      // extract bitstream meta data to figure out whether a bit-stream can be
      // decoded
      nvjpegJpegStreamParseHeader(
          nvjpeg_handle,
          encoded_images[i].data_ptr<uint8_t>(),
          encoded_images[i].numel(),
          jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(
          nvjpeg_handle, jpeg_streams[0], &isSupported);

      if (isSupported == 0) {
        hw_input_buffer.push_back(encoded_images[i].data_ptr<uint8_t>());
        hw_input_buffer_size.push_back(encoded_images[i].numel());
        hw_output_buffer.push_back(decoded_imgs_buf[i]);
      } else {
        sw_input_buffer.push_back(encoded_images[i].data_ptr<uint8_t>());
        sw_input_buffer_size.push_back(encoded_images[i].numel());
        sw_output_buffer.push_back(decoded_imgs_buf[i]);
      }
    }
  } else {
    for (std::vector<at::Tensor>::size_type i = 0; i < encoded_images.size();
         ++i) {
      sw_input_buffer.push_back(encoded_images[i].data_ptr<uint8_t>());
      sw_input_buffer_size.push_back(encoded_images[i].numel());
      sw_output_buffer.push_back(decoded_imgs_buf[i]);
    }
  }

  if (hw_input_buffer.size() > 0) {
    // UNCHANGED behaves weird, so we use RGB instead
    status = nvjpegDecodeBatchedInitialize(
        nvjpeg_handle,
        nvjpeg_state,
        hw_input_buffer.size(),
        1,
        output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                                 : output_format);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize batch decoding: ",
        status);

    status = nvjpegDecodeBatched(
        nvjpeg_handle,
        nvjpeg_state,
        hw_input_buffer.data(),
        hw_input_buffer_size.data(),
        hw_output_buffer.data(),
        stream);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to decode batch: ", status);
  }

  if (sw_input_buffer.size() > 0) {
    status =
        nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to attach device buffer: ",
        status);
    int buffer_index = 0;
    // UNCHANGED behaves weird, so we use RGB instead
    status = nvjpegDecodeParamsSetOutputFormat(
        nvjpeg_decode_params,
        output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                                 : output_format);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to set output format: ",
        status);
    for (std::vector<at::Tensor>::size_type i = 0; i < sw_input_buffer.size();
         ++i) {
      status = nvjpegJpegStreamParse(
          nvjpeg_handle,
          sw_input_buffer[i],
          sw_input_buffer_size[i],
          0,
          0,
          jpeg_streams[buffer_index]);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to parse jpeg stream: ",
          status);

      status = nvjpegStateAttachPinnedBuffer(
          nvjpeg_decoupled_state, pinned_buffers[buffer_index]);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to attach pinned buffer: ",
          status);

      status = nvjpegDecodeJpegHost(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          nvjpeg_decode_params,
          jpeg_streams[buffer_index]);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to decode jpeg stream: ",
          status);

      cudaStatus = cudaStreamSynchronize(stream);
      STD_TORCH_CHECK(
          cudaStatus == cudaSuccess,
          "Failed to synchronize CUDA stream: ",
          cudaStatus);

      status = nvjpegDecodeJpegTransferToDevice(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          jpeg_streams[buffer_index],
          stream);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to transfer jpeg to device: ",
          status);

      buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode
                                       // to avoid an extra sync

      status = nvjpegDecodeJpegDevice(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          &sw_output_buffer[i],
          stream);
      STD_TORCH_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to decode jpeg stream: ",
          status);
    }
  }

  cudaStatus = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cudaStatus);

  // prune extraneous channels from single channel images
  if (output_format == NVJPEG_OUTPUT_UNCHANGED) {
    for (std::vector<at::Tensor>::size_type i = 0; i < output_tensors.size();
         ++i) {
      if (channels[i] == 1) {
        output_tensors[i] = output_tensors[i][0].unsqueeze(0).clone();
      }
    }
  }

  return output_tensors;
}

} // namespace image
} // namespace vision

#elif ROCJPEG_FOUND

#include <ATen/cuda/CUDAContext.h>

namespace vision {
namespace image {

namespace {
constexpr uint32_t kRocJpegPitchAlignment = 16;

uint32_t align_up(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}
} // namespace

RocJpegDecoder::RocJpegDecoder(const torch::Device& target_device)
    : target_device{target_device} {
  int device_id = target_device.has_index() ? target_device.index()
                                            : c10::cuda::current_device();
  CHECK_HIP(hipSetDevice(device_id));
  CHECK_ROCJPEG(
      rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, device_id, &rocjpeg_handle));
}

RocJpegDecoder::~RocJpegDecoder() {
  rocJpegDestroy(rocjpeg_handle);
  for (auto stream_handle : rocjpeg_stream_handles) {
    rocJpegStreamDestroy(stream_handle);
  }
}

void RocJpegDecoder::ensure_stream_handles(std::size_t num_handles) {
  while (rocjpeg_stream_handles.size() < num_handles) {
    RocJpegStreamHandle stream_handle;
    CHECK_ROCJPEG(rocJpegStreamCreate(&stream_handle));
    rocjpeg_stream_handles.push_back(stream_handle);
  }
}

std::vector<torch::Tensor> RocJpegDecoder::decode_images(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode) {
  RocJpegOutputFormat output_format;
  switch (mode) {
    case vision::image::IMAGE_READ_MODE_UNCHANGED:
      output_format = ROCJPEG_OUTPUT_NATIVE;
      break;
    case vision::image::IMAGE_READ_MODE_GRAY:
      output_format = ROCJPEG_OUTPUT_Y;
      break;
    case vision::image::IMAGE_READ_MODE_RGB:
      output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
      break;
    default:
      STD_TORCH_CHECK(
          false, "The provided mode is not supported for JPEG decoding on GPU");
  }

  const std::size_t num_images = encoded_images.size();
  ensure_stream_handles(num_images);

  std::vector<RocJpegDecodeParams> decode_params(num_images);
  std::vector<RocJpegImage> output_images(num_images);
  std::vector<torch::Tensor> output_tensors(num_images);
  std::vector<int> source_channels(num_images);

  for (std::size_t i = 0; i < num_images; ++i) {
    CHECK_ROCJPEG(rocJpegStreamParse(
        static_cast<const unsigned char*>(encoded_images[i].data_ptr()),
        encoded_images[i].numel(),
        rocjpeg_stream_handles[i]));

    uint8_t num_components = 0;
    RocJpegChromaSubsampling subsampling = ROCJPEG_CSS_UNKNOWN;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    CHECK_ROCJPEG(rocJpegGetImageInfo(
        rocjpeg_handle,
        rocjpeg_stream_handles[i],
        &num_components,
        &subsampling,
        widths,
        heights));

    const uint32_t width = widths[0];
    const uint32_t height = heights[0];
    STD_TORCH_CHECK(
        width >= 64 && height >= 64,
        "Image resolution ",
        width,
        "x",
        height,
        " is below the VCN hardware JPEG decoder minimum of 64x64");
    STD_TORCH_CHECK(
        subsampling != ROCJPEG_CSS_411 && subsampling != ROCJPEG_CSS_UNKNOWN,
        "The image chroma subsampling is not supported by the VCN hardware JPEG decoder");

    // VCN writes rows at a 16-byte-aligned pitch, so allocate a buffer padded
    // to that alignment and return a view of the valid region.
    uint32_t pitch = align_up(width, kRocJpegPitchAlignment);
    uint32_t num_channels;
    switch (output_format) {
      case ROCJPEG_OUTPUT_NATIVE:
        switch (subsampling) {
          case ROCJPEG_CSS_444:
          case ROCJPEG_CSS_440:
          case ROCJPEG_CSS_420:
            num_channels = 3;
            break;
          case ROCJPEG_CSS_422:
            num_channels = 1;
            pitch = align_up(width * 2, kRocJpegPitchAlignment);
            break;
          case ROCJPEG_CSS_400:
            num_channels = 1;
            break;
          default:
            TORCH_CHECK(false, "Unsupported rocJPEG native chroma subsampling");
        }
        break;
      case ROCJPEG_OUTPUT_Y:
        num_channels = 1;
        break;
      case ROCJPEG_OUTPUT_RGB_PLANAR:
        num_channels = 3;
        break;
      default:
        TORCH_CHECK(false, "Unsupported rocJPEG output format");
    }

    auto buffer = torch::empty(
        {int64_t(num_channels),
         int64_t(align_up(height, kRocJpegPitchAlignment)),
         int64_t(pitch)},
        torch::dtype(torch::kU8).device(target_device));

    auto image_output_format = output_format;
    if (output_format == ROCJPEG_OUTPUT_NATIVE) {
      // ROCJPEG_OUTPUT_NATIVE returns YUV/native layouts whose channel count
      // and plane sizes depend on chroma subsampling. torchvision's UNCHANGED
      // mode is expected to match the CPU/nvJPEG behavior: grayscale JPEGs
      // return one channel, while color JPEGs return RGB. Decode to that
      // compatible layout.
      image_output_format =
          num_components == 1 ? ROCJPEG_OUTPUT_Y : ROCJPEG_OUTPUT_RGB_PLANAR;
    }
    decode_params[i].output_format = image_output_format;
    for (uint32_t c = 0; c < num_channels; ++c) {
      output_images[i].channel[c] = buffer[c].data_ptr<uint8_t>();
      output_images[i].pitch[c] = pitch;
    }
    source_channels[i] = num_components;
    output_tensors[i] = buffer.narrow(1, 0, height).narrow(2, 0, width);
  }

  // Choosing a batch size that is a multiple of the available JPEG cores is
  // recommended.
  CHECK_ROCJPEG(rocJpegDecodeBatched(
      rocjpeg_handle,
      rocjpeg_stream_handles.data(),
      static_cast<int>(num_images),
      decode_params.data(),
      output_images.data()));

  for (std::size_t i = 0; i < num_images; ++i) {
    output_tensors[i] = output_tensors[i].contiguous();
  }

  return output_tensors;
}

} // namespace image
} // namespace vision

#endif
