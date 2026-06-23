#include "decode_jpegs_cuda.h"

#if ROCJPEG_FOUND

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>

namespace vision {
namespace image {

namespace {
uint32_t align_up(uint32_t value) {
  constexpr uint32_t kRocJpegPitchAlignment = 16;
  return (value + kRocJpegPitchAlignment - 1) & ~(kRocJpegPitchAlignment - 1);
}

std::mutex decoderMutex;
std::unique_ptr<RocJpegDecoder> rocJpegDecoder;
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
    // rocJPEG requires contiguous input; contiguous() is a no-op when it
    // already is.
    contig_images.push_back(encoded_image.contiguous());
  }

  at::cuda::CUDAGuard device_guard(device);

  if (rocJpegDecoder == nullptr || device != rocJpegDecoder->target_device) {
    if (rocJpegDecoder != nullptr) {
      rocJpegDecoder.reset(new RocJpegDecoder(device));
    } else {
      rocJpegDecoder = std::make_unique<RocJpegDecoder>(device);
      std::atexit([]() { rocJpegDecoder.reset(); });
    }
  }

  try {
    return rocJpegDecoder->decode_images(contig_images, mode);
  } catch (const std::exception& e) {
    STD_TORCH_CHECK(false, "Error while decoding JPEG images: ", e.what());
  }
}

RocJpegDecoder::RocJpegDecoder(const torch::Device& target_device)
    : target_device{target_device} {
  int device_id = target_device.has_index() ? target_device.index()
                                            : c10::cuda::current_device();
  CHECK_HIP(hipSetDevice(device_id));
  CHECK_ROCJPEG(
      rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, device_id, &rocjpeg_handle_));
}

RocJpegDecoder::~RocJpegDecoder() {
  rocJpegDestroy(rocjpeg_handle_);
  for (auto stream_handle : rocjpeg_stream_handles_) {
    rocJpegStreamDestroy(stream_handle);
  }
}

std::vector<torch::Tensor> RocJpegDecoder::decode_images(
    const std::vector<torch::Tensor>& encoded_images,
    vision::image::ImageReadMode mode) {
  const std::size_t num_images = encoded_images.size();
  // Reuse existing rocJPEG stream handles and create only the missing ones.
  while (rocjpeg_stream_handles_.size() < num_images) {
    RocJpegStreamHandle stream_handle;
    CHECK_ROCJPEG(rocJpegStreamCreate(&stream_handle));
    rocjpeg_stream_handles_.push_back(stream_handle);
  }

  std::vector<RocJpegDecodeParams> decode_params(num_images);
  std::vector<RocJpegImage> output_images(num_images);
  std::vector<torch::Tensor> output_tensors(num_images);

  for (std::size_t i = 0; i < num_images; ++i) {
    CHECK_ROCJPEG(rocJpegStreamParse(
        static_cast<const unsigned char*>(encoded_images[i].data_ptr()),
        encoded_images[i].numel(),
        rocjpeg_stream_handles_[i]));

    uint8_t num_components = 0;
    RocJpegChromaSubsampling subsampling = ROCJPEG_CSS_UNKNOWN;
    uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    CHECK_ROCJPEG(rocJpegGetImageInfo(
        rocjpeg_handle_,
        rocjpeg_stream_handles_[i],
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
        " is below the rocJPEG hardware JPEG decoder minimum of 64x64");
    STD_TORCH_CHECK(
        subsampling != ROCJPEG_CSS_411 && subsampling != ROCJPEG_CSS_UNKNOWN,
        "The image chroma subsampling is not supported by the rocJPEG hardware JPEG decoder");

    RocJpegOutputFormat image_output_format;
    uint32_t num_channels;
    switch (mode) {
      case vision::image::IMAGE_READ_MODE_UNCHANGED:
        // torchvision's UNCHANGED mode is expected to match the CPU/nvJPEG
        // behavior: grayscale JPEGs return one channel, while color JPEGs
        // return RGB.
        if (num_components == 1) {
          image_output_format = ROCJPEG_OUTPUT_Y;
          num_channels = 1;
        } else {
          image_output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
          num_channels = 3;
        }
        break;
      case vision::image::IMAGE_READ_MODE_GRAY:
        image_output_format = ROCJPEG_OUTPUT_Y;
        num_channels = 1;
        break;
      case vision::image::IMAGE_READ_MODE_RGB:
        image_output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
        num_channels = 3;
        break;
      default:
        STD_TORCH_CHECK(
            false,
            "The provided mode is not supported for JPEG decoding on GPU");
    }

    // rocJPEG writes rows at a 16-byte-aligned pitch, so allocate a buffer
    // padded to that alignment and return a view of the valid region.
    uint32_t pitch = align_up(width);
    auto buffer = torch::empty(
        {int64_t(num_channels), int64_t(align_up(height)), int64_t(pitch)},
        torch::dtype(torch::kU8).device(target_device));

    decode_params[i].output_format = image_output_format;
    for (uint32_t c = 0; c < num_channels; ++c) {
      output_images[i].channel[c] = buffer[c].data_ptr<uint8_t>();
      output_images[i].pitch[c] = pitch;
    }
    output_tensors[i] = buffer.narrow(1, 0, height).narrow(2, 0, width);
  }

  // Choosing a batch size that is a multiple of the available JPEG cores is
  // recommended.
  CHECK_ROCJPEG(rocJpegDecodeBatched(
      rocjpeg_handle_,
      rocjpeg_stream_handles_.data(),
      static_cast<int>(num_images),
      decode_params.data(),
      output_images.data()));
  // rocJpegDecodeBatched synchronizes rocJPEG's internal HIP stream before
  // returning, so the decoded output tensors are ready for PyTorch streams.

  return output_tensors;
}

} // namespace image
} // namespace vision

#endif
