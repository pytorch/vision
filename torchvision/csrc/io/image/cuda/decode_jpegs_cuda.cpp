#include "decode_jpegs_cuda.h"
#if !NVJPEG_FOUND
namespace vision {
namespace image {

using namespace vision::stable;

std::vector<Tensor> decode_jpegs_cuda(
    const std::vector<Tensor>& encoded_images,
    vision::image::ImageReadMode mode,
    Device device) {
  VISION_CHECK(
      false, "decode_jpegs_cuda: torchvision not compiled with nvJPEG support");
}
} // namespace image
} // namespace vision

#else
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <typeinfo>
namespace vision {
namespace image {

using namespace vision::stable;

std::mutex decoderMutex;
std::unique_ptr<CUDAJpegDecoder> cudaJpegDecoder;

std::vector<Tensor> decode_jpegs_cuda(
    const std::vector<Tensor>& encoded_images,
    vision::image::ImageReadMode mode,
    Device device) {
  // Note: C10_LOG_API_USAGE_ONCE is not available in stable ABI

  std::lock_guard<std::mutex> lock(decoderMutex);
  std::vector<Tensor> contig_images;
  contig_images.reserve(encoded_images.size());

  VISION_CHECK(
      device.is_cuda(), "Expected the device parameter to be a cuda device");

  for (auto& encoded_image : encoded_images) {
    VISION_CHECK(
        encoded_image.scalar_type() == kByte, "Expected a torch.uint8 tensor");

    VISION_CHECK(
        !encoded_image.is_cuda(),
        "The input tensor must be on CPU when decoding with nvjpeg")

    VISION_CHECK(
        encoded_image.dim() == 1 && encoded_image.numel() > 0,
        "Expected a non empty 1-dimensional tensor");

    // nvjpeg requires images to be contiguous
    if (is_contiguous(encoded_image)) {
      contig_images.push_back(encoded_image);
    } else {
      contig_images.push_back(torch::stable::contiguous(encoded_image));
    }
  }

  int major_version;
  int minor_version;
  nvjpegStatus_t get_major_property_status =
      nvjpegGetProperty(MAJOR_VERSION, &major_version);
  nvjpegStatus_t get_minor_property_status =
      nvjpegGetProperty(MINOR_VERSION, &minor_version);

  VISION_CHECK(
      get_major_property_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegGetProperty failed: ",
      get_major_property_status);
  VISION_CHECK(
      get_minor_property_status == NVJPEG_STATUS_SUCCESS,
      "nvjpegGetProperty failed: ",
      get_minor_property_status);
  // Note: TORCH_WARN_ONCE is not available in stable ABI, so we skip the
  // warning about nvjpeg memory leaks in CUDA versions < 11.6

  at::cuda::CUDAGuard device_guard(at::Device(at::kCUDA, device.index()));

  if (cudaJpegDecoder == nullptr || device != cudaJpegDecoder->target_device) {
    if (cudaJpegDecoder != nullptr) {
      cudaJpegDecoder.reset(new CUDAJpegDecoder(device));
    } else {
      cudaJpegDecoder = std::make_unique<CUDAJpegDecoder>(device);
      std::atexit([]() { cudaJpegDecoder.reset(); });
    }
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
      VISION_CHECK(
          false, "The provided mode is not supported for JPEG decoding on GPU");
  }

  try {
    at::cuda::CUDAEvent event;
    auto result = cudaJpegDecoder->decode_images(contig_images, output_format);
    auto current_stream{
        device.has_index() ? at::cuda::getCurrentCUDAStream(
                                 cudaJpegDecoder->original_device.index())
                           : at::cuda::getCurrentCUDAStream()};
    event.record(cudaJpegDecoder->stream);
    event.block(current_stream);
    return result;
  } catch (const std::exception& e) {
    if (typeid(e) != typeid(std::runtime_error)) {
      VISION_CHECK(false, "Error while decoding JPEG images: ", e.what());
    } else {
      throw;
    }
  }
}

CUDAJpegDecoder::CUDAJpegDecoder(const Device& target_device)
    : original_device{kCUDA, c10::cuda::current_device()},
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
    VISION_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with default backend: ",
        status);
    hw_decode_available = false;
  } else {
    VISION_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with hardware backend: ",
        status);
  }

  status = nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg state: ",
      status);

  status = nvjpegDecoderCreate(
      nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder: ",
      status);

  status = nvjpegDecoderStateCreate(
      nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decoder state: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[0]);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferPinnedCreate(nvjpeg_handle, NULL, &pinned_buffers[1]);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create pinned buffer: ",
      status);

  status = nvjpegBufferDeviceCreate(nvjpeg_handle, NULL, &device_buffer);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create device buffer: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[0]);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_streams[1]);
  VISION_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create jpeg stream: ",
      status);

  status = nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params);
  VISION_CHECK(
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
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg decode params: ",
  //     status);

  // status = nvjpegJpegStreamDestroy(jpeg_streams[0]);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy jpeg stream: ",
  //     status);

  // status = nvjpegJpegStreamDestroy(jpeg_streams[1]);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy jpeg stream: ",
  //     status);

  // status = nvjpegBufferPinnedDestroy(pinned_buffers[0]);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy pinned buffer[0]: ",
  //     status);

  // status = nvjpegBufferPinnedDestroy(pinned_buffers[1]);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy pinned buffer[1]: ",
  //     status);

  // status = nvjpegBufferDeviceDestroy(device_buffer);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy device buffer: ",
  //     status);

  // status = nvjpegJpegStateDestroy(nvjpeg_decoupled_state);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg decoupled state: ",
  //     status);

  // status = nvjpegDecoderDestroy(nvjpeg_decoder);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg decoder: ",
  //     status);

  // status = nvjpegJpegStateDestroy(nvjpeg_state);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS,
  //     "Failed to destroy nvjpeg state: ",
  //     status);

  // status = nvjpegDestroy(nvjpeg_handle);
  // TORCH_CHECK(
  //     status == NVJPEG_STATUS_SUCCESS, "nvjpegDestroy failed: ", status);
}

std::tuple<std::vector<nvjpegImage_t>, std::vector<Tensor>, std::vector<int>>
CUDAJpegDecoder::prepare_buffers(
    const std::vector<Tensor>& encoded_images,
    const nvjpegOutputFormat_t& output_format) {
  /*
    This function scans the encoded images' jpeg headers and
    allocates decoding buffers based on the metadata found

    Args:
    - encoded_images (std::vector<Tensor>): a vector of tensors
    containing the jpeg bitstreams to be decoded. Each tensor must have dtype
    torch.uint8 and device cpu
    - output_format (nvjpegOutputFormat_t): NVJPEG_OUTPUT_RGB, NVJPEG_OUTPUT_Y
    or NVJPEG_OUTPUT_UNCHANGED

    Returns:
    - decoded_images (std::vector<nvjpegImage_t>): a vector of nvjpegImages
    containing pointers to the memory of the decoded images
    - output_tensors (std::vector<Tensor>): a vector of Tensors
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

  std::vector<Tensor> output_tensors{encoded_images.size()};
  std::vector<nvjpegImage_t> decoded_images{encoded_images.size()};

  for (size_t i = 0; i < encoded_images.size(); i++) {
    // extract bitstream meta data to figure out the number of channels, height,
    // width for every image
    status = nvjpegGetImageInfo(
        nvjpeg_handle,
        encoded_images[i].const_data_ptr<uint8_t>(),
        encoded_images[i].numel(),
        &channels[i],
        &subsampling,
        width,
        height);
    VISION_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to get image info: ", status);

    VISION_CHECK(
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
    auto output_tensor = empty(
        {int64_t(output_channels), int64_t(height[0]), int64_t(width[0])},
        kByte,
        target_device);
    output_tensors[i] = output_tensor;

    // fill nvjpegImage_t struct
    for (int c = 0; c < output_channels; c++) {
      decoded_images[i].channel[c] = torch::stable::select(output_tensor, 0, c)
                                         .mutable_data_ptr<uint8_t>();
      decoded_images[i].pitch[c] = width[0];
    }
    for (int c = output_channels; c < NVJPEG_MAX_COMPONENT; c++) {
      decoded_images[i].channel[c] = NULL;
      decoded_images[i].pitch[c] = 0;
    }
  }
  return {decoded_images, output_tensors, channels};
}

std::vector<Tensor> CUDAJpegDecoder::decode_images(
    const std::vector<Tensor>& encoded_images,
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
    - encoded_images (std::vector<Tensor>): a vector of tensors
    containing the jpeg bitstreams to be decoded
    - output_format (nvjpegOutputFormat_t): NVJPEG_OUTPUT_RGB, NVJPEG_OUTPUT_Y
    or NVJPEG_OUTPUT_UNCHANGED
    - device (Device): The desired CUDA device for the returned Tensors

    Returns:
    - output_tensors (std::vector<Tensor>): a vector of Tensors
    containing the decoded images
  */

  auto [decoded_imgs_buf, output_tensors, channels] =
      prepare_buffers(encoded_images, output_format);

  nvjpegStatus_t status;
  cudaError_t cudaStatus;

  cudaStatus = cudaStreamSynchronize(stream);
  VISION_CHECK(
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
    for (size_t i = 0; i < encoded_images.size(); ++i) {
      // extract bitstream meta data to figure out whether a bit-stream can be
      // decoded
      nvjpegJpegStreamParseHeader(
          nvjpeg_handle,
          encoded_images[i].const_data_ptr<uint8_t>(),
          encoded_images[i].numel(),
          jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(
          nvjpeg_handle, jpeg_streams[0], &isSupported);

      if (isSupported == 0) {
        hw_input_buffer.push_back(encoded_images[i].const_data_ptr<uint8_t>());
        hw_input_buffer_size.push_back(encoded_images[i].numel());
        hw_output_buffer.push_back(decoded_imgs_buf[i]);
      } else {
        sw_input_buffer.push_back(encoded_images[i].const_data_ptr<uint8_t>());
        sw_input_buffer_size.push_back(encoded_images[i].numel());
        sw_output_buffer.push_back(decoded_imgs_buf[i]);
      }
    }
  } else {
    for (size_t i = 0; i < encoded_images.size(); ++i) {
      sw_input_buffer.push_back(encoded_images[i].const_data_ptr<uint8_t>());
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
    VISION_CHECK(
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
    VISION_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to decode batch: ", status);
  }

  if (sw_input_buffer.size() > 0) {
    status =
        nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer);
    VISION_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to attach device buffer: ",
        status);
    int buffer_index = 0;
    // UNCHANGED behaves weird, so we use RGB instead
    status = nvjpegDecodeParamsSetOutputFormat(
        nvjpeg_decode_params,
        output_format == NVJPEG_OUTPUT_UNCHANGED ? NVJPEG_OUTPUT_RGB
                                                 : output_format);
    VISION_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to set output format: ",
        status);
    for (size_t i = 0; i < sw_input_buffer.size(); ++i) {
      status = nvjpegJpegStreamParse(
          nvjpeg_handle,
          sw_input_buffer[i],
          sw_input_buffer_size[i],
          0,
          0,
          jpeg_streams[buffer_index]);
      VISION_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to parse jpeg stream: ",
          status);

      status = nvjpegStateAttachPinnedBuffer(
          nvjpeg_decoupled_state, pinned_buffers[buffer_index]);
      VISION_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to attach pinned buffer: ",
          status);

      status = nvjpegDecodeJpegHost(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          nvjpeg_decode_params,
          jpeg_streams[buffer_index]);
      VISION_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to decode jpeg stream: ",
          status);

      cudaStatus = cudaStreamSynchronize(stream);
      VISION_CHECK(
          cudaStatus == cudaSuccess,
          "Failed to synchronize CUDA stream: ",
          cudaStatus);

      status = nvjpegDecodeJpegTransferToDevice(
          nvjpeg_handle,
          nvjpeg_decoder,
          nvjpeg_decoupled_state,
          jpeg_streams[buffer_index],
          stream);
      VISION_CHECK(
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
      VISION_CHECK(
          status == NVJPEG_STATUS_SUCCESS,
          "Failed to decode jpeg stream: ",
          status);
    }
  }

  cudaStatus = cudaStreamSynchronize(stream);
  VISION_CHECK(
      cudaStatus == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cudaStatus);

  // prune extraneous channels from single channel images
  if (output_format == NVJPEG_OUTPUT_UNCHANGED) {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      if (channels[i] == 1) {
        output_tensors[i] = torch::stable::clone(torch::stable::unsqueeze(
            torch::stable::select(output_tensors[i], 0, 0), 0));
      }
    }
  }

  return output_tensors;
}

} // namespace image
} // namespace vision

#endif
