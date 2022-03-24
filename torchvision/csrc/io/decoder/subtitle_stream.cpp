#include "subtitle_stream.h"
#include <c10/util/Logging.h>
#include <limits>
#include "util.h"

namespace ffmpeg {
const AVRational timeBaseQ = AVRational{1, AV_TIME_BASE};

SubtitleStream::SubtitleStream(
    AVFormatContext* inputCtx,
    int index,
    bool convertPtsToWallTime,
    const SubtitleFormat& format)
    : Stream(
          inputCtx,
          MediaFormat::makeMediaFormat(format, index),
          convertPtsToWallTime,
          0) {
  memset(&sub_, 0, sizeof(sub_));
}

void SubtitleStream::releaseSubtitle() {
  if (sub_.release) {
    avsubtitle_free(&sub_);
    memset(&sub_, 0, sizeof(sub_));
  }
}

SubtitleStream::~SubtitleStream() {
  releaseSubtitle();
  sampler_.shutdown();
}

int SubtitleStream::initFormat() {
  if (!codecCtx_->subtitle_header) {
    LOG(ERROR) << "No subtitle header found";
  } else {
    VLOG(1) << "Subtitle header found!";
  }
  return 0;
}

int SubtitleStream::analyzePacket(const AVPacket* packet, bool* gotFrame) {
  // clean-up
  releaseSubtitle();

  // FIXME: should this even be created?
  AVPacket* avPacket;
  avPacket = av_packet_alloc();
  if (avPacket == nullptr) {
    LOG(ERROR)
        << "decoder as not able to allocate the subtitle-specific packet.";
    // alternative to ENOMEM
    return AVERROR_BUFFER_TOO_SMALL;
  }
  avPacket->data = nullptr;
  avPacket->size = 0;
  // check flush packet
  auto pkt = packet ? packet : avPacket;

  int gotFramePtr = 0;
  // is these a better way than cast from const?
  int result =
      avcodec_decode_subtitle2(codecCtx_, &sub_, &gotFramePtr, (AVPacket*)pkt);

  if (result < 0) {
    LOG(ERROR) << "avcodec_decode_subtitle2 failed, err: "
               << Util::generateErrorDesc(result);
    // free the packet we've created
    av_packet_free(&avPacket);
    return result;
  } else if (result == 0) {
    result = pkt->size; // discard the rest of the package
  }

  sub_.release = gotFramePtr;
  *gotFrame = gotFramePtr > 0;

  // set proper pts in us
  if (gotFramePtr) {
    sub_.pts = av_rescale_q(
        pkt->pts, inputCtx_->streams[format_.stream]->time_base, timeBaseQ);
  }

  av_packet_free(&avPacket);
  return result;
}

int SubtitleStream::copyFrameBytes(ByteStorage* out, bool flush) {
  return sampler_.sample(flush ? nullptr : &sub_, out);
}

void SubtitleStream::setFramePts(DecoderHeader* header, bool) {
  header->pts = sub_.pts; // already in us
}

} // namespace ffmpeg
