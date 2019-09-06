#include "FfmpegStream.h"
#include "FfmpegUtil.h"

using namespace std;

// (TODO) Currently, disable the use of refCount
static int refCount = 0;

FfmpegStream::FfmpegStream(
    AVFormatContext* inputCtx,
    int index,
    enum AVMediaType avMediaType,
    double seekFrameMargin)
    : inputCtx_(inputCtx),
      index_(index),
      avMediaType_(avMediaType),
      seekFrameMargin_(seekFrameMargin) {
}

FfmpegStream::~FfmpegStream() {
  if (frame_) {
    av_frame_free(&frame_);
  }
  avcodec_free_context(&codecCtx_);
}

int FfmpegStream::openCodecContext() {
  VLOG(2) << "stream start_time: " << inputCtx_->streams[index_]->start_time;

  auto typeString = av_get_media_type_string(avMediaType_);
  AVStream* st = inputCtx_->streams[index_];
  auto codec_id = st->codecpar->codec_id;
  VLOG(1) << "codec_id: " << codec_id;
  AVCodec* codec = avcodec_find_decoder(codec_id);
  if (!codec) {
    LOG(ERROR) << "avcodec_find_decoder failed for codec_id: " << int(codec_id);
    return AVERROR(EINVAL);
  }
  VLOG(1) << "Succeed to find decoder";

  codecCtx_ = avcodec_alloc_context3(codec);
  if (!codecCtx_) {
    LOG(ERROR) << "avcodec_alloc_context3 fails";
    return AVERROR(ENOMEM);
  }

  int ret;
  /* Copy codec parameters from input stream to output codec context */
  if ((ret = avcodec_parameters_to_context(codecCtx_, st->codecpar)) < 0) {
    LOG(ERROR) << "Failed to copy " << typeString
               << " codec parameters to decoder context";
    return ret;
  }

  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "refcounted_frames", refCount ? "1" : "0", 0);

  // after avcodec_open2, value of codecCtx_->time_base is NOT meaningful
  // But inputCtx_->streams[index_]->time_base has meaningful values
  if ((ret = avcodec_open2(codecCtx_, codec, &opts)) < 0) {
    LOG(ERROR) << "avcodec_open2 failed. " << ffmpeg_util::getErrorDesc(ret);
    return ret;
  }
  VLOG(1) << "Succeed to open codec";

  frame_ = av_frame_alloc();
  return initFormat();
}

unique_ptr<DecodedFrame> FfmpegStream::getFrameData(int getPtsOnly) {
  if (!codecCtx_) {
    LOG(ERROR) << "Codec is not initialized";
    return nullptr;
  }
  if (getPtsOnly) {
    unique_ptr<DecodedFrame> decodedFrame = make_unique<DecodedFrame>();
    decodedFrame->pts_ = frame_->pts;
    return decodedFrame;
  } else {
    unique_ptr<DecodedFrame> decodedFrame = sampleFrameData();
    if (decodedFrame) {
      decodedFrame->pts_ = frame_->pts;
    }
    return decodedFrame;
  }
}

<<<<<<< Updated upstream
<<<<<<< HEAD
void FfmpegStream::flush(int getPtsOnly, DecoderOutput& decoderOutput) {
  VLOG(1) << "flush stream";
=======
void FfmpegStream::flush(int getPtsOnly, DecoderOutput& decoderOutput)
{
  VLOG(1) << "Media Type: " << getMediaType() << ", flush stream.";
>>>>>>> 73af0db8f428e9137ee85db70a1532a96ca407d5
=======
void FfmpegStream::flush(int getPtsOnly, DecoderOutput& decoderOutput)
{
  VLOG(1) << "Media Type: " << getMediaType() << ", flush stream.";
>>>>>>> Stashed changes
  // need to receive frames before entering draining mode
  receiveAvailFrames(getPtsOnly, decoderOutput);

  VLOG(2) << "send nullptr packet";
  sendPacket(nullptr);
  // receive remaining frames after entering draining mode
  receiveAvailFrames(getPtsOnly, decoderOutput);

  avcodec_flush_buffers(codecCtx_);
}

bool FfmpegStream::isFramePtsInRange() {
  CHECK(frame_);
  auto pts = frame_->pts;
  auto startPts = this->getStartPts();
  auto endPts = this->getEndPts();
<<<<<<< Updated upstream
<<<<<<< HEAD
  VLOG(2) << "isPtsInRange. pts: " << pts << ", startPts: " << startPts
          << ", endPts: " << endPts;
  return (pts == AV_NOPTS_VALUE) ||
      (pts >= startPts && (endPts > 0 ? pts <= endPts : true));
=======
  VLOG(2) << "isPtsInRange. pts: " << pts
          << ", startPts: " << startPts
          << ", endPts: " << endPts;
  return (pts == AV_NOPTS_VALUE) || (pts >= startPts && (endPts >= 0 ? pts <= endPts : true));
>>>>>>> 73af0db8f428e9137ee85db70a1532a96ca407d5
=======
  VLOG(2) << "isPtsInRange. pts: " << pts
          << ", startPts: " << startPts
          << ", endPts: " << endPts;
  return (pts == AV_NOPTS_VALUE) || (pts >= startPts && (endPts >= 0 ? pts <= endPts : true));
>>>>>>> Stashed changes
}

bool FfmpegStream::isFramePtsExceedRange() {
  if (frame_) {
    auto endPts = this->getEndPts();
<<<<<<< Updated upstream
<<<<<<< HEAD
    VLOG(2) << "isFramePtsExceedRange. last_pts_: " << last_pts_
            << ", endPts: " << endPts;
    return endPts > 0 ? last_pts_ >= endPts : false;
=======
    VLOG(2) << "isFramePtsExceedRange. last_pts_: " << last_pts_ << ", endPts: " << endPts;
    return endPts >= 0 ? last_pts_ >= endPts : false;
>>>>>>> 73af0db8f428e9137ee85db70a1532a96ca407d5
=======
    VLOG(2) << "isFramePtsExceedRange. last_pts_: " << last_pts_ << ", endPts: " << endPts;
    return endPts >= 0 ? last_pts_ >= endPts : false;
>>>>>>> Stashed changes
  } else {
    return true;
  }
}

// seek a frame
int FfmpegStream::seekFrame(int64_t seekPts) {
  // translate margin from second to pts
  int64_t margin = (int64_t)(
      seekFrameMargin_ * (double)inputCtx_->streams[index_]->time_base.den /
      (double)inputCtx_->streams[index_]->time_base.num);
  int64_t real_seekPts = (seekPts - margin) > 0 ? (seekPts - margin) : 0;
  VLOG(2) << "seek margin: " << margin;
  VLOG(2) << "real seekPts: " << real_seekPts;
  int ret = av_seek_frame(
      inputCtx_,
      index_,
      (seekPts - margin) > 0 ? (seekPts - margin) : 0,
      AVSEEK_FLAG_BACKWARD);
  if (ret < 0) {
    LOG(WARNING) << "av_seek_frame fails. Stream index: " << index_;
    return ret;
  }
  return 0;
}

<<<<<<< Updated upstream
<<<<<<< HEAD
=======

>>>>>>> 73af0db8f428e9137ee85db70a1532a96ca407d5
=======

>>>>>>> Stashed changes
// send/receive encoding and decoding API overview
// https://ffmpeg.org/doxygen/3.4/group__lavc__encdec.html
int FfmpegStream::sendPacket(const AVPacket* packet) {
  return avcodec_send_packet(codecCtx_, packet);
}

int FfmpegStream::receiveFrame() {
  int ret = avcodec_receive_frame(codecCtx_, frame_);
  if (ret >= 0) {
    // succeed
    frame_->pts = av_frame_get_best_effort_timestamp(frame_);
    if (frame_->pts == AV_NOPTS_VALUE) {
<<<<<<< Updated upstream
<<<<<<< HEAD
      // Trick: if we can not figure out pts, we just set it to be (last_pts +
      // 1)
=======
      // Trick: if we can not figure out pts, we just set it to be (last_pts + 1)
>>>>>>> 73af0db8f428e9137ee85db70a1532a96ca407d5
=======
      // Trick: if we can not figure out pts, we just set it to be (last_pts + 1)
>>>>>>> Stashed changes
      frame_->pts = last_pts_ + 1;
    }
    last_pts_ = frame_->pts;

    VLOG(2) << "avcodec_receive_frame succeed";
  } else if (ret == AVERROR(EAGAIN)) {
    VLOG(2) << "avcodec_receive_frame fails and returns AVERROR(EAGAIN). ";
  } else if (ret == AVERROR_EOF) {
    // no more frame to read
    VLOG(2) << "avcodec_receive_frame returns AVERROR_EOF";
  } else {
<<<<<<< Updated upstream
<<<<<<< HEAD
    LOG(WARNING) << "avcodec_receive_frame failed. Error: "
                 << ffmpeg_util::getErrorDesc(ret);
=======
    LOG(WARNING) << "avcodec_receive_frame failed. Error: " << ffmpeg_util::getErrorDesc(ret);
>>>>>>> 73af0db8f428e9137ee85db70a1532a96ca407d5
=======
    LOG(WARNING) << "avcodec_receive_frame failed. Error: " << ffmpeg_util::getErrorDesc(ret);
>>>>>>> Stashed changes
  }
  return ret;
}

<<<<<<< Updated upstream
<<<<<<< HEAD
void FfmpegStream::receiveAvailFrames(
    int getPtsOnly,
    DecoderOutput& decoderOutput) {
=======
void FfmpegStream::receiveAvailFrames(int getPtsOnly, DecoderOutput& decoderOutput) {
>>>>>>> 73af0db8f428e9137ee85db70a1532a96ca407d5
=======
void FfmpegStream::receiveAvailFrames(int getPtsOnly, DecoderOutput& decoderOutput) {
>>>>>>> Stashed changes
  int result = 0;
  while ((result = receiveFrame()) >= 0) {
    unique_ptr<DecodedFrame> decodedFrame = getFrameData(getPtsOnly);

<<<<<<< Updated upstream
<<<<<<< HEAD
    if (decodedFrame &&
        ((!getPtsOnly && decodedFrame->frameSize_ > 0) || getPtsOnly)) {
=======
    if (decodedFrame && ((!getPtsOnly && decodedFrame->frameSize_ > 0) || getPtsOnly))
    {
>>>>>>> 73af0db8f428e9137ee85db70a1532a96ca407d5
=======
    if (decodedFrame && ((!getPtsOnly && decodedFrame->frameSize_ > 0) || getPtsOnly))
    {
>>>>>>> Stashed changes
      if (isFramePtsInRange()) {
        decoderOutput.addMediaFrame(getMediaType(), std::move(decodedFrame));
      }
    } // end-if
  } // end-while
}
