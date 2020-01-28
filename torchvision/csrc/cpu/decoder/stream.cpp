// Copyright 2004-present Facebook. All Rights Reserved.

#include "stream.h"
#include <c10/util/Logging.h>
#include "util.h"

namespace ffmpeg {

namespace {
const size_t kDecoderHeaderSize = sizeof(DecoderHeader);
}

Stream::Stream(
    AVFormatContext* inputCtx,
    MediaFormat format,
    bool convertPtsToWallTime)
    : inputCtx_(inputCtx),
      format_(format),
      convertPtsToWallTime_(convertPtsToWallTime) {}

Stream::~Stream() {
  if (frame_) {
    av_free(frame_); // Copyright 2004-present Facebook. All Rights Reserved.
  }
  if (codecCtx_) {
    avcodec_free_context(&codecCtx_);
  }
}

AVCodec* Stream::findCodec(AVCodecContext* ctx) {
  return avcodec_find_decoder(ctx->codec_id);
}

int Stream::openCodec() {
  AVStream* steam = inputCtx_->streams[format_.stream];
  auto codec_id = steam->codecpar->codec_id;
  AVCodec* codec = avcodec_find_decoder(codec_id);
  if (!codec) {
    LOG(ERROR) << "avcodec_find_decoder failed for codec_id: " << int(codec_id);
    return AVERROR(EINVAL);
  }

  if (!(codecCtx_ = avcodec_alloc_context3(codec))) {
    LOG(ERROR) << "avcodec_alloc_context3 fails";
    return AVERROR(ENOMEM);
  }

  int ret;
  // Copy codec parameters from input stream to output codec context
  if ((ret = avcodec_parameters_to_context(codecCtx_, steam->codecpar)) < 0) {
    LOG(ERROR) << "Failed to copy codec parameters to decoder context";
    return ret;
  }

  // after avcodec_open2, value of codecCtx_->time_base is NOT meaningful
  if ((ret = avcodec_open2(codecCtx_, codec, nullptr)) < 0) {
    LOG(ERROR) << "avcodec_open2 failed. " << Util::generateErrorDesc(ret);
    avcodec_free_context(&codecCtx_);
    codecCtx_ = nullptr;
    return ret;
  }

  frame_ = av_frame_alloc();

  format_.num = inputCtx_->streams[format_.stream]->time_base.num;
  format_.den = inputCtx_->streams[format_.stream]->time_base.den;
  format_.duration = inputCtx_->streams[format_.stream]->duration;

  return initFormat();
}

// rescale package
void Stream::rescalePackage(AVPacket* packet) {
  if (codecCtx_->time_base.num != 0) {
    av_packet_rescale_ts(
        packet,
        inputCtx_->streams[format_.stream]->time_base,
        codecCtx_->time_base);
  }
}

int Stream::analyzePacket(const AVPacket* packet, int* gotFramePtr) {
  int consumed = 0;
  int result = avcodec_send_packet(codecCtx_, packet);
  if (result == AVERROR(EAGAIN)) {
    *gotFramePtr = 0; // no bytes get consumed, fetch frame
  } else if (result == AVERROR_EOF) {
    *gotFramePtr = 0; // more than one flush packet
    if (packet) {
      // got packet after flush, this is an error
      return result;
    }
  } else if (result < 0) {
    LOG(ERROR) << "avcodec_send_packet failed, err: "
               << Util::generateErrorDesc(result);
    return result; // error
  } else {
    consumed = packet ? packet->size : 0; // all bytes get consumed
  }

  result = avcodec_receive_frame(codecCtx_, frame_);

  if (result >= 0) {
    *gotFramePtr = 1; // frame is available
  } else if (result == AVERROR(EAGAIN)) {
    *gotFramePtr = 0; // no frames at this time, needs more packets
    if (!consumed) {
      // precaution, if no packages got consumed and no frames are available
      return result;
    }
  } else if (result == AVERROR_EOF) {
    *gotFramePtr = 0; // the last frame has been flushed
    // precaution, if no more frames are available assume we consume all bytes
    consumed = packet ? packet->size : 0;
  } else { // error
    LOG(ERROR) << "avcodec_receive_frame failed, err: "
               << Util::generateErrorDesc(result);
    return result;
  }
  return consumed;
}

int Stream::decodeFrame(const AVPacket* packet, int* gotFramePtr) {
  return analyzePacket(packet, gotFramePtr);
}

int Stream::getFrameBytes(DecoderOutputMessage* out, bool headerOnly) {
  return fillBuffer(out, false, headerOnly);
}

int Stream::flush(DecoderOutputMessage* out, bool headerOnly) {
  int gotFramePtr = 0;
  int result;
  if (analyzePacket(nullptr, &gotFramePtr) >= 0 && gotFramePtr &&
      (result = fillBuffer(out, false, headerOnly)) > 0) {
    return result;
  } else if ((result = fillBuffer(out, true, headerOnly)) > 0) {
    return result;
  }
  return result;
}

int Stream::fillBuffer(DecoderOutputMessage* out, bool flush, bool headerOnly) {
  int result = -1;
  if (!codecCtx_) {
    LOG(INFO) << "Codec is not initialized";
    return result;
  }

  // assign message
  setHeader(&out->header);

  if (headerOnly) {
    return sizeof(out->header);
  }

  // init sampler, if any and return required bytes
  if ((result = estimateBytes(flush)) < 0) {
    return result;
  }
  out->payload->ensure(result);
  return copyFrameBytes(out->payload.get(), flush);
}

} // namespace ffmpeg
