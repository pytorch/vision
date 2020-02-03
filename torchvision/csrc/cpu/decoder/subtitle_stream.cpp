#include "subtitle_stream.h"
#include <c10/util/Logging.h>
#include <limits>
#include "util.h"

namespace ffmpeg {

namespace {

bool operator==(const SubtitleFormat&, const AVCodecContext&) {
  return true;
}

SubtitleFormat& toSubtitleFormat(SubtitleFormat& x, const AVCodecContext&) {
  return x;
}
} // namespace

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
    LOG(INFO) << "Subtitle header found!";
  }
  return 0;
}

int SubtitleStream::analyzePacket(const AVPacket* packet, bool* gotFrame) {
  // clean-up
  releaseSubtitle();
  // check flush packet
  AVPacket avPacket;
  av_init_packet(&avPacket);
  avPacket.data = nullptr;
  auto pkt = packet ? *packet : avPacket;
  int gotFramePtr = 0;
  int result = avcodec_decode_subtitle2(codecCtx_, &sub_, &gotFramePtr, &pkt);

  if (result < 0) {
    VLOG(1) << "avcodec_decode_subtitle2 failed, err: "
            << Util::generateErrorDesc(result);
  } else if (result == 0) {
    result = packet ? packet->size : 0; // discard the rest of the package
  }

  sub_.release = gotFramePtr;
  *gotFrame = gotFramePtr > 0;
  return result;
}

int SubtitleStream::estimateBytes(bool) {
  if (!(sampler_.getInputFormat().subtitle == *codecCtx_)) {
    // - reinit sampler
    SamplerParameters params;
    params.type = MediaType::TYPE_SUBTITLE;
    toSubtitleFormat(params.in.subtitle, *codecCtx_);
    if (!sampler_.init(params)) {
      return -1;
    }

    VLOG(1) << "Set input subtitle sampler format";
  }
  return sampler_.getSamplesBytes(&sub_);
}

int SubtitleStream::copyFrameBytes(ByteStorage* out, bool flush) {
  return sampler_.sample(flush ? nullptr : &sub_, out);
}

void SubtitleStream::setFramePts(DecoderHeader* header, bool) {
  header->pts = sub_.pts; // already in us
}

} // namespace ffmpeg
