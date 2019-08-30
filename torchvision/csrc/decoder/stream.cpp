// Copyright 2004-present Facebook. All Rights Reserved.

#include "stream.h"
#include <glog/logging.h>
#include "util.h"

namespace ffmpeg {

namespace {
const size_t kDecoderHeaderSize = sizeof(DecoderHeader);
}

Stream::Stream(AVFormatContext* inputCtx, int index, bool convertPtsToWallTime)
    : inputCtx_(inputCtx),
      index_(index),
      convertPtsToWallTime_(convertPtsToWallTime) {}

Stream::~Stream() {
  if (frame_) {
    av_free(frame_); // Copyright 2004-present Facebook. All Rights Reserved.
  }
  if (codecCtx_) {
    avcodec_close(codecCtx_);
  }
}

AVCodec* Stream::findCodec(AVCodecContext* ctx) {
  return avcodec_find_decoder(ctx->codec_id);
}

int Stream::openCodec() {
  auto codecCtx = inputCtx_->streams[index_]->codec;
  AVCodec* codec = findCodec(codecCtx);

  if (!codec) {
    LOG(ERROR) << "findCodec failed for codec_id: " << int(codecCtx->codec_id);
    return -1;
  }

  auto result = avcodec_open2(codecCtx, codec, nullptr);
  if (result < 0) {
    LOG(ERROR) << "avcodec_open2 failed, error: "
               << Util::generateErrorDesc(result);
    return result;
  } else {
    VLOG(1) << "avcodec_open2 opened codec id: " << codecCtx->codec_id;
  }

  codecCtx_ = codecCtx;

  frame_ = av_frame_alloc();

  return initFormat();
}

// rescale package
void Stream::rescalePackage(AVPacket* packet) {
  if (codecCtx_->time_base.num != 0) {
    av_packet_rescale_ts(
        packet, inputCtx_->streams[index_]->time_base, codecCtx_->time_base);
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
      // got packet after flush, this is error
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

int Stream::getFrameBytes(DecoderOutputMessage* out) {
  return fillBuffer(out, false);
}

int Stream::flush(DecoderOutputMessage* out) {
  int gotFramePtr = 0;
  int result = -1;
  if (analyzePacket(nullptr, &gotFramePtr) >= 0 && gotFramePtr &&
      (result = fillBuffer(out, false)) > 0) {
    return result;
  } else if ((result = fillBuffer(out, true)) > 0) {
    return result;
  }
  return result;
}

int Stream::fillBuffer(DecoderOutputMessage* out, bool flush) {
  int result = -1;
  if (!codecCtx_) {
    LOG(INFO) << "Codec is not initialized";
    return result;
  }

  // init sampler, if any and return required bytes
  if ((result = estimateBytes(flush)) < 0) {
    return result;
  }
  // assign message
  setHeader(&out->header);
  out->payload->ensure(result);
  return copyFrameBytes(out->payload.get(), flush);
}

} // namespace ffmpeg
