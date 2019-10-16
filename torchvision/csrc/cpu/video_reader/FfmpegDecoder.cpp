#include "FfmpegDecoder.h"
#include "FfmpegAudioStream.h"
#include "FfmpegUtil.h"
#include "FfmpegVideoStream.h"

using namespace std;

static AVPacket avPkt;

namespace {

unique_ptr<FfmpegStream> createFfmpegStream(
    MediaType type,
    AVFormatContext* ctx,
    int idx,
    MediaFormat& mediaFormat,
    double seekFrameMargin) {
  enum AVMediaType avType;
  CHECK(ffmpeg_util::mapMediaType(type, &avType));
  switch (type) {
    case MediaType::TYPE_VIDEO:
      return make_unique<FfmpegVideoStream>(
          ctx, idx, avType, mediaFormat, seekFrameMargin);
    case MediaType::TYPE_AUDIO:
      return make_unique<FfmpegAudioStream>(
          ctx, idx, avType, mediaFormat, seekFrameMargin);
    default:
      return nullptr;
  }
}

} // namespace

FfmpegAvioContext::FfmpegAvioContext()
    : workBuffersize_(VIO_BUFFER_SZ),
      workBuffer_((uint8_t*)av_malloc(workBuffersize_)),
      inputFile_(nullptr),
      inputBuffer_(nullptr),
      inputBufferSize_(0) {}

int FfmpegAvioContext::initAVIOContext(const uint8_t* buffer, int64_t size) {
  inputBuffer_ = buffer;
  inputBufferSize_ = size;
  avioCtx_ = avio_alloc_context(
      workBuffer_,
      workBuffersize_,
      0,
      reinterpret_cast<void*>(this),
      &FfmpegAvioContext::readMemory,
      nullptr, // no write function
      &FfmpegAvioContext::seekMemory);
  return 0;
}

FfmpegAvioContext::~FfmpegAvioContext() {
  /* note: the internal buffer could have changed, and be != workBuffer_ */
  if (avioCtx_) {
    av_freep(&avioCtx_->buffer);
    av_freep(&avioCtx_);
  } else {
    av_freep(&workBuffer_);
  }
  if (inputFile_) {
    fclose(inputFile_);
  }
}

int FfmpegAvioContext::read(uint8_t* buf, int buf_size) {
  if (inputBuffer_) {
    return readMemory(this, buf, buf_size);
  } else {
    return -1;
  }
}

int FfmpegAvioContext::readMemory(void* opaque, uint8_t* buf, int buf_size) {
  FfmpegAvioContext* h = static_cast<FfmpegAvioContext*>(opaque);
  if (buf_size < 0) {
    return -1;
  }

  int reminder = h->inputBufferSize_ - h->offset_;
  int r = buf_size < reminder ? buf_size : reminder;
  if (r < 0) {
    return AVERROR_EOF;
  }

  memcpy(buf, h->inputBuffer_ + h->offset_, r);
  h->offset_ += r;
  return r;
}

int64_t FfmpegAvioContext::seek(int64_t offset, int whence) {
  if (inputBuffer_) {
    return seekMemory(this, offset, whence);
  } else {
    return -1;
  }
}

int64_t FfmpegAvioContext::seekMemory(
    void* opaque,
    int64_t offset,
    int whence) {
  FfmpegAvioContext* h = static_cast<FfmpegAvioContext*>(opaque);
  switch (whence) {
    case SEEK_CUR: // from current position
      h->offset_ += offset;
      break;
    case SEEK_END: // from eof
      h->offset_ = h->inputBufferSize_ + offset;
      break;
    case SEEK_SET: // from beginning of file
      h->offset_ = offset;
      break;
    case AVSEEK_SIZE:
      return h->inputBufferSize_;
  }
  return h->offset_;
}

int FfmpegDecoder::init(
    const std::string& filename,
    bool isDecodeFile,
    FfmpegAvioContext& ioctx,
    DecoderOutput& decoderOutput) {
  cleanUp();

  int ret = 0;
  if (!isDecodeFile) {
    formatCtx_ = avformat_alloc_context();
    if (!formatCtx_) {
      LOG(ERROR) << "avformat_alloc_context failed";
      return -1;
    }
    formatCtx_->pb = ioctx.get_avio();
    formatCtx_->flags |= AVFMT_FLAG_CUSTOM_IO;

    // Determining the input format:
    int probeSz = AVPROBE_SIZE + AVPROBE_PADDING_SIZE;
    uint8_t* probe((uint8_t*)av_malloc(probeSz));
    memset(probe, 0, probeSz);
    int len = ioctx.read(probe, probeSz - AVPROBE_PADDING_SIZE);
    if (len < probeSz - AVPROBE_PADDING_SIZE) {
      LOG(ERROR) << "Insufficient data to determine video format";
      av_freep(&probe);
      return -1;
    }
    // seek back to start of stream
    ioctx.seek(0, SEEK_SET);

    unique_ptr<AVProbeData> probeData(new AVProbeData());
    probeData->buf = probe;
    probeData->buf_size = len;
    probeData->filename = "";
    // Determine the input-format:
    formatCtx_->iformat = av_probe_input_format(probeData.get(), 1);
    // this is to avoid the double-free error
    if (formatCtx_->iformat == nullptr) {
      LOG(ERROR) << "av_probe_input_format fails";
      return -1;
    }
    VLOG(1) << "av_probe_input_format succeeds";
    av_freep(&probe);

    ret = avformat_open_input(&formatCtx_, "", nullptr, nullptr);
  } else {
    ret = avformat_open_input(&formatCtx_, filename.c_str(), nullptr, nullptr);
  }

  if (ret < 0) {
    LOG(ERROR) << "avformat_open_input failed, error: "
               << ffmpeg_util::getErrorDesc(ret);
    cleanUp();
    return ret;
  }
  ret = avformat_find_stream_info(formatCtx_, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "avformat_find_stream_info failed, error: "
               << ffmpeg_util::getErrorDesc(ret);
    cleanUp();
    return ret;
  }
  if (!initStreams()) {
    LOG(ERROR) << "Cannot activate streams";
    cleanUp();
    return -1;
  }

  for (auto& stream : streams_) {
    MediaType mediaType = stream.second->getMediaType();
    decoderOutput.initMediaType(mediaType, stream.second->getMediaFormat());
  }
  VLOG(1) << "FfmpegDecoder initialized";
  return 0;
}

int FfmpegDecoder::decodeFile(
    unique_ptr<DecoderParameters> params,
    const string& fileName,
    DecoderOutput& decoderOutput) {
  VLOG(1) << "decode file: " << fileName;
  FfmpegAvioContext ioctx;
  int ret = decodeLoop(std::move(params), fileName, true, ioctx, decoderOutput);
  return ret;
}

int FfmpegDecoder::decodeMemory(
    unique_ptr<DecoderParameters> params,
    const uint8_t* buffer,
    int64_t size,
    DecoderOutput& decoderOutput) {
  VLOG(1) << "decode video data in memory";
  FfmpegAvioContext ioctx;
  int ret = ioctx.initAVIOContext(buffer, size);
  if (ret == 0) {
    ret =
        decodeLoop(std::move(params), string(""), false, ioctx, decoderOutput);
  }
  return ret;
}

int FfmpegDecoder::probeFile(
    unique_ptr<DecoderParameters> params,
    const string& fileName,
    DecoderOutput& decoderOutput) {
  VLOG(1) << "probe file: " << fileName;
  FfmpegAvioContext ioctx;
  return probeVideo(std::move(params), fileName, true, ioctx, decoderOutput);
}

int FfmpegDecoder::probeMemory(
    unique_ptr<DecoderParameters> params,
    const uint8_t* buffer,
    int64_t size,
    DecoderOutput& decoderOutput) {
  VLOG(1) << "probe video data in memory";
  FfmpegAvioContext ioctx;
  int ret = ioctx.initAVIOContext(buffer, size);
  if (ret == 0) {
    ret =
        probeVideo(std::move(params), string(""), false, ioctx, decoderOutput);
  }
  return ret;
}

void FfmpegDecoder::cleanUp() {
  if (formatCtx_) {
    for (auto& stream : streams_) {
      // Drain stream buffers.
      DecoderOutput decoderOutput;
      stream.second->flush(1, decoderOutput);
      stream.second.reset();
    }
    streams_.clear();
    avformat_close_input(&formatCtx_);
  }
}

FfmpegStream* FfmpegDecoder::findStreamByIndex(int streamIndex) const {
  auto it = streams_.find(streamIndex);
  return it != streams_.end() ? it->second.get() : nullptr;
}

/*
Reference implementation:
https://ffmpeg.org/doxygen/3.4/demuxing_decoding_8c-example.html
*/
int FfmpegDecoder::decodeLoop(
    unique_ptr<DecoderParameters> params,
    const std::string& filename,
    bool isDecodeFile,
    FfmpegAvioContext& ioctx,
    DecoderOutput& decoderOutput) {
  params_ = std::move(params);

  int ret = init(filename, isDecodeFile, ioctx, decoderOutput);
  if (ret < 0) {
    return ret;
  }
  // init package
  av_init_packet(&avPkt);
  avPkt.data = nullptr;
  avPkt.size = 0;

  int result = 0;
  bool ptsInRange = true;
  while (ptsInRange) {
    result = av_read_frame(formatCtx_, &avPkt);
    if (result == AVERROR(EAGAIN)) {
      VLOG(1) << "Decoder is busy";
      ret = 0;
      break;
    } else if (result == AVERROR_EOF) {
      VLOG(1) << "Stream decoding is completed";
      ret = 0;
      break;
    } else if (result < 0) {
      VLOG(1) << "av_read_frame fails. Break decoder loop. Error: "
              << ffmpeg_util::getErrorDesc(result);
      ret = result;
      break;
    }

    ret = 0;
    auto stream = findStreamByIndex(avPkt.stream_index);
    if (stream == nullptr) {
      // the packet is from a stream the caller is not interested. Ignore it
      VLOG(2) << "avPkt ignored. stream index: " << avPkt.stream_index;
      // Need to free the memory of AVPacket. Otherwise, memory leak happens
      av_packet_unref(&avPkt);
      continue;
    }

    do {
      result = stream->sendPacket(&avPkt);
      if (result == AVERROR(EAGAIN)) {
        VLOG(2) << "avcodec_send_packet returns AVERROR(EAGAIN)";
        // start to recevie available frames from internal buffer
        stream->receiveAvailFrames(params_->getPtsOnly, decoderOutput);
        if (isPtsExceedRange()) {
          // exit the most-outer while loop
          VLOG(1) << "In all streams, exceed the end pts. Exit decoding loop";
          ret = 0;
          ptsInRange = false;
          break;
        }
      } else if (result < 0) {
        LOG(WARNING) << "avcodec_send_packet failed. Error: "
                     << ffmpeg_util::getErrorDesc(result);
        ret = result;
        break;
      } else {
        VLOG(2) << "avcodec_send_packet succeeds";
        // succeed. Read the next AVPacket and send out it
        break;
      }
    } while (ptsInRange);
    // Need to free the memory of AVPacket. Otherwise, memory leak happens
    av_packet_unref(&avPkt);
  }
  /* flush cached frames */
  flushStreams(decoderOutput);
  return ret;
}

int FfmpegDecoder::probeVideo(
    unique_ptr<DecoderParameters> params,
    const std::string& filename,
    bool isDecodeFile,
    FfmpegAvioContext& ioctx,
    DecoderOutput& decoderOutput) {
  params_ = std::move(params);
  return init(filename, isDecodeFile, ioctx, decoderOutput);
}

bool FfmpegDecoder::initStreams() {
  for (auto it = params_->formats.begin(); it != params_->formats.end(); ++it) {
    AVMediaType mediaType;
    if (!ffmpeg_util::mapMediaType(it->first, &mediaType)) {
      LOG(ERROR) << "Unknown media type: " << it->first;
      return false;
    }
    int streamIdx =
        av_find_best_stream(formatCtx_, mediaType, -1, -1, nullptr, 0);

    if (streamIdx >= 0) {
      VLOG(2) << "find stream index: " << streamIdx;
      auto stream = createFfmpegStream(
          it->first,
          formatCtx_,
          streamIdx,
          it->second,
          params_->seekFrameMargin);

      CHECK(stream);
      if (stream->openCodecContext() < 0) {
        LOG(ERROR) << "Cannot open codec. Stream index: " << streamIdx;
        return false;
      }
      streams_.emplace(streamIdx, move(stream));
    } else {
      VLOG(1) << "Cannot open find stream of type " << it->first;
    }
  }
  // Seek frames in each stream
  int ret = 0;
  for (auto& stream : streams_) {
    auto startPts = stream.second->getStartPts();
    VLOG(1) << "stream: " << stream.first << " startPts: " << startPts;
    if (startPts > 0 && (ret = stream.second->seekFrame(startPts)) < 0) {
      LOG(WARNING) << "seekFrame in stream fails";
      return false;
    }
  }
  VLOG(1) << "initStreams succeeds";
  return true;
}

bool FfmpegDecoder::isPtsExceedRange() {
  bool exceed = true;
  for (auto& stream : streams_) {
    exceed = exceed && stream.second->isFramePtsExceedRange();
  }
  return exceed;
}

void FfmpegDecoder::flushStreams(DecoderOutput& decoderOutput) {
  for (auto& stream : streams_) {
    stream.second->flush(params_->getPtsOnly, decoderOutput);
  }
}
