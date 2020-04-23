#include "stream.h"
#include <c10/util/Logging.h>
#include "util.h"

namespace ffmpeg {

Stream::Stream(
    AVFormatContext* inputCtx,
    MediaFormat format,
    bool convertPtsToWallTime,
    int64_t loggingUuid)
    : inputCtx_(inputCtx),
      format_(format),
      convertPtsToWallTime_(convertPtsToWallTime),
      loggingUuid_(loggingUuid) {}

Stream::~Stream() {
  if (frame_) {
    av_free(frame_); // Copyright 2004-present Facebook. All Rights Reserved.
  }
  if (codecCtx_) {
    avcodec_free_context(&codecCtx_);
  }
}

AVCodec* Stream::findCodec(AVCodecParameters* params) {
  return avcodec_find_decoder(params->codec_id);
}

int Stream::openCodec(std::vector<DecoderMetadata>* metadata) {
  AVStream* steam = inputCtx_->streams[format_.stream];

  AVCodec* codec = findCodec(steam->codecpar);
  if (!codec) {
    LOG(ERROR) << "LoggingUuid #" << loggingUuid_
               << ", avcodec_find_decoder failed for codec_id: "
               << int(steam->codecpar->codec_id);
    return AVERROR(EINVAL);
  }

  if (!(codecCtx_ = avcodec_alloc_context3(codec))) {
    LOG(ERROR) << "LoggingUuid #" << loggingUuid_
               << ", avcodec_alloc_context3 failed";
    return AVERROR(ENOMEM);
  }

  int ret;
  // Copy codec parameters from input stream to output codec context
  if ((ret = avcodec_parameters_to_context(codecCtx_, steam->codecpar)) < 0) {
    LOG(ERROR) << "LoggingUuid #" << loggingUuid_
               << ", avcodec_parameters_to_context failed";
    return ret;
  }

  // after avcodec_open2, value of codecCtx_->time_base is NOT meaningful
  if ((ret = avcodec_open2(codecCtx_, codec, nullptr)) < 0) {
    LOG(ERROR) << "LoggingUuid #" << loggingUuid_
               << ", avcodec_open2 failed: " << Util::generateErrorDesc(ret);
    avcodec_free_context(&codecCtx_);
    codecCtx_ = nullptr;
    return ret;
  }

  frame_ = av_frame_alloc();

  switch (format_.type) {
    case TYPE_VIDEO:
      fps_ = av_q2d(av_guess_frame_rate(inputCtx_, steam, nullptr));
      break;
    case TYPE_AUDIO:
      fps_ = codecCtx_->sample_rate;
      break;
    default:
      fps_ = 30.0;
  }

  if ((ret = initFormat())) {
    LOG(ERROR) << "initFormat failed, type: " << format_.type;
  }

  if (metadata) {
    DecoderMetadata header;
    header.format = format_;
    header.fps = fps_;
    header.num = steam->time_base.num;
    header.den = steam->time_base.den;
    header.duration =
        av_rescale_q(steam->duration, steam->time_base, AV_TIME_BASE_Q);
    metadata->push_back(header);
  }

  return ret;
}

int Stream::analyzePacket(const AVPacket* packet, bool* gotFrame) {
  int consumed = 0;
  int result = avcodec_send_packet(codecCtx_, packet);
  if (result == AVERROR(EAGAIN)) {
    *gotFrame = false; // no bytes get consumed, fetch frame
  } else if (result == AVERROR_EOF) {
    *gotFrame = false; // more than one flush packet
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
    *gotFrame = true; // frame is available
  } else if (result == AVERROR(EAGAIN)) {
    *gotFrame = false; // no frames at this time, needs more packets
    if (!consumed) {
      // precaution, if no packages got consumed and no frames are available
      return result;
    }
  } else if (result == AVERROR_EOF) {
    *gotFrame = false; // the last frame has been flushed
    // precaution, if no more frames are available assume we consume all bytes
    consumed = 0;
  } else { // error
    LOG(ERROR) << "avcodec_receive_frame failed, err: "
               << Util::generateErrorDesc(result);
    return result;
  }
  return consumed;
}

int Stream::decodePacket(
    const AVPacket* packet,
    DecoderOutputMessage* out,
    bool headerOnly,
    bool* hasMsg) {
  int consumed;
  bool gotFrame = false;
  *hasMsg = false;
  if ((consumed = analyzePacket(packet, &gotFrame)) >= 0 &&
      (packet == nullptr || gotFrame)) {
    int result;
    if ((result = getMessage(out, !gotFrame, headerOnly)) < 0) {
      return result; // report error
    }
    *hasMsg = result > 0;
  }
  return consumed;
}

int Stream::flush(DecoderOutputMessage* out, bool headerOnly) {
  bool hasMsg = false;
  int result = decodePacket(nullptr, out, headerOnly, &hasMsg);
  if (result < 0) {
    avcodec_flush_buffers(codecCtx_);
    return result;
  }
  if (!hasMsg) {
    avcodec_flush_buffers(codecCtx_);
    return 0;
  }
  return 1;
}

int Stream::getMessage(DecoderOutputMessage* out, bool flush, bool headerOnly) {
  if (flush) {
    // only flush of audio frames makes sense
    if (format_.type == TYPE_AUDIO) {
      int processed = 0;
      size_t total = 0;
      // grab all audio bytes by chunks
      do {
        if ((processed = copyFrameBytes(out->payload.get(), flush)) < 0) {
          return processed;
        }
        total += processed;
      } while (processed);

      if (total) {
        // set header if message bytes are available
        setHeader(&out->header, flush);
        return 1;
      }
    }
    return 0;
  } else {
    if (format_.type == TYPE_AUDIO) {
      int processed = 0;
      if ((processed = copyFrameBytes(out->payload.get(), flush)) < 0) {
        return processed;
      }
      if (processed) {
        // set header if message bytes are available
        setHeader(&out->header, flush);
        return 1;
      }
      return 0;
    } else {
      // set header
      setHeader(&out->header, flush);

      if (headerOnly) {
        // Only header is requisted
        return 1;
      }

      return copyFrameBytes(out->payload.get(), flush);
    }
  }
}

void Stream::setHeader(DecoderHeader* header, bool flush) {
  header->seqno = numGenerator_++;

  setFramePts(header, flush);

  if (convertPtsToWallTime_) {
    keeper_.adjust(header->pts);
  }

  header->format = format_;
  header->keyFrame = 0;
  header->fps = std::numeric_limits<double>::quiet_NaN();
}

void Stream::setFramePts(DecoderHeader* header, bool flush) {
  if (flush) {
    header->pts = nextPts_; // already in us
  } else {
    header->pts = av_frame_get_best_effort_timestamp(frame_);
    if (header->pts == AV_NOPTS_VALUE) {
      header->pts = nextPts_;
    } else {
      header->pts = av_rescale_q(
          header->pts,
          inputCtx_->streams[format_.stream]->time_base,
          AV_TIME_BASE_Q);
    }

    switch (format_.type) {
      case TYPE_AUDIO:
        nextPts_ = header->pts + frame_->nb_samples * AV_TIME_BASE / fps_;
        break;
      case TYPE_VIDEO:
        nextPts_ = header->pts + AV_TIME_BASE / fps_;
        break;
      default:
        nextPts_ = header->pts;
    }
  }
}

} // namespace ffmpeg
