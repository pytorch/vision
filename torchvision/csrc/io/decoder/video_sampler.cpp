#include "video_sampler.h"
#include <c10/util/Logging.h>
#include "util.h"

// www.ffmpeg.org/doxygen/0.5/swscale-example_8c-source.html

namespace ffmpeg {

namespace {

// Setup the data pointers and linesizes based on the specified image
// parameters and the provided array. This sets up "planes" to point to a
// "buffer"
// NOTE: this is most likely culprit behind #3534
//
// Args:
// fmt: desired output video format
// buffer: source constant image buffer (in different format) that will contain
// the final image after SWScale planes: destination data pointer to be filled
// lineSize: target destination linesize (always {0})
int preparePlanes(
    const VideoFormat& fmt,
    const uint8_t* buffer,
    uint8_t** planes,
    int* lineSize) {
  int result;

  // NOTE: 1 at the end of av_fill_arrays is the value used for alignment
  if ((result = av_image_fill_arrays(
           planes,
           lineSize,
           buffer,
           (AVPixelFormat)fmt.format,
           fmt.width,
           fmt.height,
           1)) < 0) {
    LOG(ERROR) << "av_image_fill_arrays failed, err: "
               << Util::generateErrorDesc(result);
  }
  return result;
}

// Scale (and crop) the image slice in srcSlice and put the resulting scaled
// slice to `planes` buffer, which is mapped to be `out` via preparePlanes as
// `sws_scale` cannot access buffers directly.
//
// Args:
// context: SWSContext allocated on line 119 (if crop, optional) or 163 (if
// scale) srcSlice: frame data in YUV420P srcStride: the array containing the
// strides for each plane of the source
//            image (from AVFrame->linesize[0])
// out: destination buffer
// planes: indirect destination buffer (mapped to "out" via preparePlanes)
// lines: destination linesize; constant {0}
int transformImage(
    SwsContext* context,
    const uint8_t* const srcSlice[],
    int srcStride[],
    VideoFormat inFormat,
    VideoFormat outFormat,
    uint8_t* out,
    uint8_t* planes[],
    int lines[]) {
  int result;
  if ((result = preparePlanes(outFormat, out, planes, lines)) < 0) {
    return result;
  }
  if (context) {
    // NOTE: srcY stride always 0: this is a parameter of YUV format
    if ((result = sws_scale(
             context, srcSlice, srcStride, 0, inFormat.height, planes, lines)) <
        0) {
      LOG(ERROR) << "sws_scale failed, err: "
                 << Util::generateErrorDesc(result);
      return result;
    }
  } else if (
      inFormat.width == outFormat.width &&
      inFormat.height == outFormat.height &&
      inFormat.format == outFormat.format) {
    // Copy planes without using sws_scale if sws_getContext failed.
    av_image_copy(
        planes,
        lines,
        (const uint8_t**)srcSlice,
        srcStride,
        (AVPixelFormat)inFormat.format,
        inFormat.width,
        inFormat.height);
  } else {
    LOG(ERROR) << "Invalid scale context format " << inFormat.format;
    return AVERROR(EINVAL);
  }
  return 0;
}
} // namespace

VideoSampler::VideoSampler(int swsFlags, int64_t loggingUuid)
    : swsFlags_(swsFlags), loggingUuid_(loggingUuid) {}

VideoSampler::~VideoSampler() {
  cleanUp();
}

void VideoSampler::shutdown() {
  cleanUp();
}

bool VideoSampler::init(const SamplerParameters& params) {
  cleanUp();

  if (params.out.video.cropImage != 0) {
    if (!Util::validateVideoFormat(params.out.video)) {
      LOG(ERROR) << "Invalid video format"
                 << ", width: " << params.out.video.width
                 << ", height: " << params.out.video.height
                 << ", format: " << params.out.video.format
                 << ", minDimension: " << params.out.video.minDimension
                 << ", crop: " << params.out.video.cropImage;

      return false;
    }

    scaleFormat_.format = params.out.video.format;
    Util::setFormatDimensions(
        scaleFormat_.width,
        scaleFormat_.height,
        params.out.video.width,
        params.out.video.height,
        params.in.video.width,
        params.in.video.height,
        0,
        0,
        1);

    if (!(scaleFormat_ == params_.out.video)) { // crop required
      cropContext_ = sws_getContext(
          params.out.video.width,
          params.out.video.height,
          (AVPixelFormat)params.out.video.format,
          params.out.video.width,
          params.out.video.height,
          (AVPixelFormat)params.out.video.format,
          swsFlags_,
          nullptr,
          nullptr,
          nullptr);

      if (!cropContext_) {
        LOG(ERROR) << "sws_getContext failed for crop context";
        return false;
      }

      const auto scaleImageSize = av_image_get_buffer_size(
          (AVPixelFormat)scaleFormat_.format,
          scaleFormat_.width,
          scaleFormat_.height,
          1);
      scaleBuffer_.resize(scaleImageSize);
    }
  } else {
    scaleFormat_ = params.out.video;
  }

  VLOG(1) << "Input format #" << loggingUuid_ << ", width "
          << params.in.video.width << ", height " << params.in.video.height
          << ", format " << params.in.video.format << ", minDimension "
          << params.in.video.minDimension << ", cropImage "
          << params.in.video.cropImage;
  VLOG(1) << "Scale format #" << loggingUuid_ << ", width "
          << scaleFormat_.width << ", height " << scaleFormat_.height
          << ", format " << scaleFormat_.format << ", minDimension "
          << scaleFormat_.minDimension << ", cropImage "
          << scaleFormat_.cropImage;
  VLOG(1) << "Crop format #" << loggingUuid_ << ", width "
          << params.out.video.width << ", height " << params.out.video.height
          << ", format " << params.out.video.format << ", minDimension "
          << params.out.video.minDimension << ", cropImage "
          << params.out.video.cropImage;

  // set output format
  params_ = params;

  scaleContext_ = sws_getContext(
      params.in.video.width,
      params.in.video.height,
      (AVPixelFormat)params.in.video.format,
      scaleFormat_.width,
      scaleFormat_.height,
      (AVPixelFormat)scaleFormat_.format,
      swsFlags_,
      nullptr,
      nullptr,
      nullptr);
  // sws_getContext might fail if in/out format == AV_PIX_FMT_PAL8 (png format)
  // Return true if input and output formats/width/height are identical
  // Check scaleContext_ for nullptr in transformImage to copy planes directly

  if (params.in.video.width == scaleFormat_.width &&
      params.in.video.height == scaleFormat_.height &&
      params.in.video.format == scaleFormat_.format) {
    return true;
  }
  return scaleContext_ != nullptr;
}

// Main body of the sample function called from one of the overloads below
//
// Args:
// srcSlice: decoded AVFrame->data perpared buffer
// srcStride: linesize (usually obtained from AVFrame->linesize)
// out: return buffer (ByteStorage*)
int VideoSampler::sample(
    const uint8_t* const srcSlice[],
    int srcStride[],
    ByteStorage* out) {
  int result;
  // scaled and cropped image
  int outImageSize = av_image_get_buffer_size(
      (AVPixelFormat)params_.out.video.format,
      params_.out.video.width,
      params_.out.video.height,
      1);

  out->ensure(outImageSize);

  uint8_t* scalePlanes[4] = {nullptr};
  int scaleLines[4] = {0};
  // perform scale first
  if ((result = transformImage(
           scaleContext_,
           srcSlice,
           srcStride,
           params_.in.video,
           scaleFormat_,
           // for crop use internal buffer
           cropContext_ ? scaleBuffer_.data() : out->writableTail(),
           scalePlanes,
           scaleLines))) {
    return result;
  }

  // is crop required?
  if (cropContext_) {
    uint8_t* cropPlanes[4] = {nullptr};
    int cropLines[4] = {0};

    if (params_.out.video.height < scaleFormat_.height) {
      // Destination image is wider of source image: cut top and bottom
      for (size_t i = 0; i < 4 && scalePlanes[i] != nullptr; ++i) {
        scalePlanes[i] += scaleLines[i] *
            (scaleFormat_.height - params_.out.video.height) / 2;
      }
    } else {
      // Source image is wider of destination image: cut sides
      for (size_t i = 0; i < 4 && scalePlanes[i] != nullptr; ++i) {
        scalePlanes[i] += scaleLines[i] *
            (scaleFormat_.width - params_.out.video.width) / 2 /
            scaleFormat_.width;
      }
    }

    // crop image
    if ((result = transformImage(
             cropContext_,
             scalePlanes,
             scaleLines,
             params_.out.video,
             params_.out.video,
             out->writableTail(),
             cropPlanes,
             cropLines))) {
      return result;
    }
  }

  out->append(outImageSize);
  return outImageSize;
}

// Call from `video_stream.cpp::114` - occurs during file reads
int VideoSampler::sample(AVFrame* frame, ByteStorage* out) {
  if (!frame) {
    return 0; // no flush for videos
  }

  return sample(frame->data, frame->linesize, out);
}

// Call from `video_stream.cpp::114` - not sure when this occurs
int VideoSampler::sample(const ByteStorage* in, ByteStorage* out) {
  if (!in) {
    return 0; // no flush for videos
  }

  int result;
  uint8_t* inPlanes[4] = {nullptr};
  int inLineSize[4] = {0};

  if ((result = preparePlanes(
           params_.in.video, in->data(), inPlanes, inLineSize)) < 0) {
    return result;
  }

  return sample(inPlanes, inLineSize, out);
}

void VideoSampler::cleanUp() {
  if (scaleContext_) {
    sws_freeContext(scaleContext_);
    scaleContext_ = nullptr;
  }
  if (cropContext_) {
    sws_freeContext(cropContext_);
    cropContext_ = nullptr;
    scaleBuffer_.clear();
  }
}

} // namespace ffmpeg
