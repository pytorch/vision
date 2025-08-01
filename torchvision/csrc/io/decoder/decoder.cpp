#include "decoder.h"
#include <c10/util/Logging.h>
#include <libavutil/avutil.h>
#include <future>
#include <iostream>
#include <mutex>
#include "audio_stream.h"
#include "cc_stream.h"
#include "subtitle_stream.h"
#include "util.h"
#include "video_stream.h"

namespace ffmpeg {

namespace {

constexpr size_t kIoBufferSize = 96 * 1024;
constexpr size_t kIoPaddingSize = AV_INPUT_BUFFER_PADDING_SIZE;
constexpr size_t kLogBufferSize = 1024;

bool mapFfmpegType(AVMediaType media, MediaType* type) {
  switch (media) {
    case AVMEDIA_TYPE_AUDIO:
      *type = TYPE_AUDIO;
      return true;
    case AVMEDIA_TYPE_VIDEO:
      *type = TYPE_VIDEO;
      return true;
    case AVMEDIA_TYPE_SUBTITLE:
      *type = TYPE_SUBTITLE;
      return true;
    case AVMEDIA_TYPE_DATA:
      *type = TYPE_CC;
      return true;
    default:
      return false;
  }
}

std::unique_ptr<Stream> createStream(
    MediaType type,
    AVFormatContext* ctx,
    int idx,
    bool convertPtsToWallTime,
    const FormatUnion& format,
    int64_t loggingUuid) {
  switch (type) {
    case TYPE_AUDIO:
      return std::make_unique<AudioStream>(
          ctx, idx, convertPtsToWallTime, format.audio);
    case TYPE_VIDEO:
      return std::make_unique<VideoStream>(
          // negative loggingUuid indicates video streams.
          ctx,
          idx,
          convertPtsToWallTime,
          format.video,
          -loggingUuid);
    case TYPE_SUBTITLE:
      return std::make_unique<SubtitleStream>(
          ctx, idx, convertPtsToWallTime, format.subtitle);
    case TYPE_CC:
      return std::make_unique<CCStream>(
          ctx, idx, convertPtsToWallTime, format.subtitle);
    default:
      return nullptr;
  }
}

} // Namespace

/* static */
void Decoder::logFunction(void* avcl, int level, const char* cfmt, va_list vl) {
  if (!avcl) {
    // Nothing can be done here
    return;
  }

  AVClass* avclass = *reinterpret_cast<AVClass**>(avcl);
  if (!avclass) {
    // Nothing can be done here
    return;
  }
  Decoder* decoder = nullptr;
  if (strcmp(avclass->class_name, "AVFormatContext") == 0) {
    AVFormatContext* context = reinterpret_cast<AVFormatContext*>(avcl);
    if (context) {
      decoder = reinterpret_cast<Decoder*>(context->opaque);
    }
  } else if (strcmp(avclass->class_name, "AVCodecContext") == 0) {
    AVCodecContext* context = reinterpret_cast<AVCodecContext*>(avcl);
    if (context) {
      decoder = reinterpret_cast<Decoder*>(context->opaque);
    }
  } else if (strcmp(avclass->class_name, "AVIOContext") == 0) {
    AVIOContext* context = reinterpret_cast<AVIOContext*>(avcl);
    // only if opaque was assigned to Decoder pointer
    if (context && context->read_packet == Decoder::readFunction) {
      decoder = reinterpret_cast<Decoder*>(context->opaque);
    }
  } else if (strcmp(avclass->class_name, "SWResampler") == 0) {
    // expect AVCodecContext as parent
    if (avclass->parent_log_context_offset) {
      AVClass** parent =
          *(AVClass***)(((uint8_t*)avcl) + avclass->parent_log_context_offset);
      AVCodecContext* context = reinterpret_cast<AVCodecContext*>(parent);
      if (context) {
        decoder = reinterpret_cast<Decoder*>(context->opaque);
      }
    }
  } else if (strcmp(avclass->class_name, "SWScaler") == 0) {
    // cannot find a way to pass context pointer through SwsContext struct
  } else {
    VLOG(2) << "Unknown context class: " << avclass->class_name;
  }

  if (decoder != nullptr && decoder->enableLogLevel(level)) {
    char buf[kLogBufferSize] = {0};
    // Format the line
    int* prefix = decoder->getPrintPrefix();
    *prefix = 1;
    av_log_format_line(avcl, level, cfmt, vl, buf, sizeof(buf) - 1, prefix);
    // pass message to the decoder instance
    std::string msg(buf);
    decoder->logCallback(level, msg);
  }
}

bool Decoder::enableLogLevel(int level) const {
  return ssize_t(level) <= params_.logLevel;
}

void Decoder::logCallback(int level, const std::string& message) {
  LOG(INFO) << "Msg, uuid=" << params_.loggingUuid << " level=" << level
            << " msg=" << message;
}

/* static */
int Decoder::shutdownFunction(void* ctx) {
  Decoder* decoder = (Decoder*)ctx;
  if (decoder == nullptr) {
    return 1;
  }
  return decoder->shutdownCallback();
}

int Decoder::shutdownCallback() {
  return interrupted_ ? 1 : 0;
}

/* static */
int Decoder::readFunction(void* opaque, uint8_t* buf, int size) {
  Decoder* decoder = reinterpret_cast<Decoder*>(opaque);
  if (decoder == nullptr) {
    return 0;
  }
  return decoder->readCallback(buf, size);
}

/* static */
int64_t Decoder::seekFunction(void* opaque, int64_t offset, int whence) {
  Decoder* decoder = reinterpret_cast<Decoder*>(opaque);
  if (decoder == nullptr) {
    return -1;
  }
  return decoder->seekCallback(offset, whence);
}

int Decoder::readCallback(uint8_t* buf, int size) {
  return seekableBuffer_.read(buf, size, params_.timeoutMs);
}

int64_t Decoder::seekCallback(int64_t offset, int whence) {
  return seekableBuffer_.seek(offset, whence, params_.timeoutMs);
}

/* static */
void Decoder::initOnce() {
  static std::once_flag flagInit;
  std::call_once(flagInit, []() {
#if LIBAVUTIL_VERSION_MAJOR < 56 // Before FFMPEG 4.0
    av_register_all();
    avcodec_register_all();
#endif
    avformat_network_init();
    av_log_set_callback(Decoder::logFunction);
    av_log_set_level(AV_LOG_ERROR);
    VLOG(1) << "Registered ffmpeg libs";
  });
}

Decoder::Decoder() {
  initOnce();
}

Decoder::~Decoder() {
  cleanUp();
}

// Initialise the format context that holds information about the container and
// fill it with minimal information about the format (codecs are not opened
// here). Function reads in information about the streams from the container
// into inputCtx and then passes it to decoder::openStreams. Finally, if seek is
// specified within the decoder parameters, it seeks into the correct frame
// (note, the seek defined here is "precise" seek).
bool Decoder::init(
    const DecoderParameters& params,
    DecoderInCallback&& in,
    std::vector<DecoderMetadata>* metadata) {
  cleanUp();

  if ((params.uri.empty() || in) && (!params.uri.empty() || !in)) {
    LOG(ERROR)
        << "uuid=" << params_.loggingUuid
        << " either external URI gets provided or explicit input callback";
    return false;
  }

  // set callback and params
  params_ = params;

  if (!(inputCtx_ = avformat_alloc_context())) {
    LOG(ERROR) << "uuid=" << params_.loggingUuid
               << " cannot allocate format context";
    return false;
  }

  AVInputFormat* fmt = nullptr;
  int result = 0;
  if (in) {
    ImageType type = ImageType::UNKNOWN;
    if ((result = seekableBuffer_.init(
             std::forward<DecoderInCallback>(in),
             params_.timeoutMs,
             params_.maxSeekableBytes,
             params_.isImage ? &type : nullptr)) < 0) {
      LOG(ERROR) << "uuid=" << params_.loggingUuid
                 << " can't initiate seekable buffer";
      cleanUp();
      return false;
    }

    if (params_.isImage) {
      const char* fmtName = "image2";
      switch (type) {
        case ImageType::JPEG:
          fmtName = "jpeg_pipe";
          break;
        case ImageType::PNG:
          fmtName = "png_pipe";
          break;
        case ImageType::TIFF:
          fmtName = "tiff_pipe";
          break;
        default:
          break;
      }

      fmt = (AVInputFormat*)av_find_input_format(fmtName);
    }

    const size_t avioCtxBufferSize = kIoBufferSize;
    uint8_t* avioCtxBuffer =
        (uint8_t*)av_malloc(avioCtxBufferSize + kIoPaddingSize);
    if (!avioCtxBuffer) {
      LOG(ERROR) << "uuid=" << params_.loggingUuid
                 << " av_malloc cannot allocate " << avioCtxBufferSize
                 << " bytes";
      cleanUp();
      return false;
    }

    if (!(avioCtx_ = avio_alloc_context(
              avioCtxBuffer,
              avioCtxBufferSize,
              0,
              reinterpret_cast<void*>(this),
              &Decoder::readFunction,
              nullptr,
              result == 1 ? &Decoder::seekFunction : nullptr))) {
      LOG(ERROR) << "uuid=" << params_.loggingUuid
                 << " avio_alloc_context failed";
      av_free(avioCtxBuffer);
      cleanUp();
      return false;
    }

    avioCtx_->max_packet_size = params.maxEncodedBufferSize;

    inputCtx_->pb = avioCtx_;
    inputCtx_->flags |= AVFMT_FLAG_CUSTOM_IO;
  }

  inputCtx_->opaque = reinterpret_cast<void*>(this);
  inputCtx_->interrupt_callback.callback = Decoder::shutdownFunction;
  inputCtx_->interrupt_callback.opaque = reinterpret_cast<void*>(this);

  // add network timeout
  inputCtx_->flags |= AVFMT_FLAG_NONBLOCK;

  AVDictionary* options = nullptr;
  if (params_.listen) {
    av_dict_set_int(&options, "listen", 1, 0);
  }
  if (params_.timeoutMs > 0) {
    av_dict_set_int(&options, "analyzeduration", params_.timeoutMs * 1000, 0);
    av_dict_set_int(&options, "stimeout", params_.timeoutMs * 1000, 0);
    av_dict_set_int(&options, "rw_timeout", params_.timeoutMs * 1000, 0);
    if (!params_.tlsCertFile.empty()) {
      av_dict_set(&options, "cert_file", params_.tlsCertFile.data(), 0);
    }
    if (!params_.tlsKeyFile.empty()) {
      av_dict_set(&options, "key_file", params_.tlsKeyFile.data(), 0);
    }
  }

  av_dict_set_int(&options, "probesize", params_.probeSize, 0);

  interrupted_ = false;

  // ffmpeg avformat_open_input call can hang if media source doesn't respond
  // set a guard for handle such situations, if requested
  std::promise<bool> p;
  std::future<bool> f = p.get_future();
  std::unique_ptr<std::thread> guard;
  if (params_.preventStaleness) {
    guard = std::make_unique<std::thread>([&f, this]() {
      auto timeout = std::chrono::milliseconds(params_.timeoutMs);
      if (std::future_status::timeout == f.wait_for(timeout)) {
        LOG(ERROR) << "uuid=" << params_.loggingUuid
                   << " cannot open stream within " << params_.timeoutMs
                   << " ms";
        interrupted_ = true;
      }
    });
  }

  if (fmt) {
    result = avformat_open_input(&inputCtx_, nullptr, fmt, &options);
  } else {
    result =
        avformat_open_input(&inputCtx_, params_.uri.c_str(), nullptr, &options);
  }

  av_dict_free(&options);

  if (guard) {
    p.set_value(true);
    guard->join();
    guard.reset();
  }

  if (result < 0 || interrupted_) {
    LOG(ERROR) << "uuid=" << params_.loggingUuid
               << " avformat_open_input failed, error="
               << Util::generateErrorDesc(result);
    cleanUp();
    return false;
  }

  result = avformat_find_stream_info(inputCtx_, nullptr);

  if (result < 0) {
    LOG(ERROR) << "uuid=" << params_.loggingUuid
               << " avformat_find_stream_info failed, error="
               << Util::generateErrorDesc(result);
    cleanUp();
    return false;
  }

  if (!openStreams(metadata)) {
    LOG(ERROR) << "uuid=" << params_.loggingUuid << " cannot activate streams";
    cleanUp();
    return false;
  }
  // SyncDecoder inherits Decoder which would override onInit.
  onInit();

  if (params.startOffset != 0) {
    auto offset = params.startOffset <= params.seekAccuracy
        ? 0
        : params.startOffset - params.seekAccuracy;

    av_seek_frame(inputCtx_, -1, offset, AVSEEK_FLAG_BACKWARD);
  }

  for (unsigned int i = 0; i < inputCtx_->nb_streams; i++) {
    if (
#if LIBAVUTIL_VERSION_MAJOR < 56 // Before FFMPEG 4.0
        inputCtx_->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO
#else // FFMPEG 4.0+
        inputCtx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO
#endif
        && inputCtx_->streams[i]->duration > 0) {
      // There is at least two 1/r_frame_rates from the frame before the last
      // one until the video duration, let's prefer to set duration after the
      // frame before the last one, but as early as possible
      double correction = 2 * inputCtx_->streams[i]->r_frame_rate.den /
              (double)inputCtx_->streams[i]->r_frame_rate.num -
          1 / (double)AV_TIME_BASE;
      videoDurationMs_ = 1000 * inputCtx_->streams[i]->duration *
              inputCtx_->streams[i]->time_base.num /
              (double)inputCtx_->streams[i]->time_base.den -
          1000 * correction;
      break;
    }
  }

  VLOG(1) << "Decoder initialized, log level: " << params_.logLevel;
  VLOG(1) << "Video duration: " << videoDurationMs_;
  return true;
}

// open appropriate CODEC for every type of stream and move it to the class
// variable `streams_` and make sure it is in range for decoding
bool Decoder::openStreams(std::vector<DecoderMetadata>* metadata) {
  for (unsigned int i = 0; i < inputCtx_->nb_streams; i++) {
    // - find the corespondent format at params_.formats set
    MediaFormat format;
#if LIBAVUTIL_VERSION_MAJOR < 56 // Before FFMPEG 4.0
    const auto media = inputCtx_->streams[i]->codec->codec_type;
#else // FFMPEG 4.0+
    const auto media = inputCtx_->streams[i]->codecpar->codec_type;
#endif
    if (!mapFfmpegType(media, &format.type)) {
      VLOG(1) << "Stream media: " << media << " at index " << i
              << " gets ignored, unknown type";

      continue; // unsupported type
    }

    // check format
    auto it = params_.formats.find(format);
    if (it == params_.formats.end()) {
      VLOG(1) << "Stream type: " << format.type << " at index: " << i
              << " gets ignored, caller is not interested";
      continue; // clients don't care about this media format
    }

    // do we have stream of this type?
    auto stream = findByType(format);

    // should we process this stream?

    if (it->stream == -2 || // all streams of this type are welcome
        (!stream && (it->stream == -1 || it->stream == i))) { // new stream
      VLOG(1) << "Stream type: " << format.type << " found, at index: " << i;
      auto stream_2 = createStream(
          format.type,
          inputCtx_,
          i,
          params_.convertPtsToWallTime,
          it->format,
          params_.loggingUuid);
      CHECK(stream_2);
      if (stream_2->openCodec(metadata, params_.numThreads) < 0) {
        LOG(ERROR) << "uuid=" << params_.loggingUuid
                   << " open codec failed, stream_idx=" << i;
        return false;
      }
      streams_.emplace(i, std::move(stream_2));
      inRange_.set(i, true);
    }
  }

  return true;
}

void Decoder::shutdown() {
  cleanUp();
}

void Decoder::interrupt() {
  interrupted_ = true;
}

void Decoder::cleanUp() {
  if (!interrupted_) {
    interrupted_ = true;
  }

  if (inputCtx_) {
    for (auto& stream : streams_) {
      // Drain stream buffers.
      DecoderOutputMessage msg;
      while (msg.payload = nullptr, stream.second->flush(&msg, true) > 0) {
      }
      stream.second.reset();
    }
    streams_.clear();
    avformat_close_input(&inputCtx_);
  }
  if (avioCtx_) {
    av_freep(&avioCtx_->buffer);
    av_freep(&avioCtx_);
  }

  // reset callback
  seekableBuffer_.shutdown();
}

// function does actual work, derived class calls it in working thread
// periodically. On success method returns 0, ENODATA on EOF, ETIMEDOUT if
// no frames got decoded in the specified timeout time, AVERROR_BUFFER_TOO_SMALL
// when unable to allocate packet and error on unrecoverable error
int Decoder::getFrame(size_t workingTimeInMs) {
  if (inRange_.none()) {
    return ENODATA;
  }
  // decode frames until cache is full and leave thread
  // once decode() method gets called and grab some bytes
  // run this method again
  // init package
  // update 03/22: moving memory management to ffmpeg
  AVPacket* avPacket;
  avPacket = av_packet_alloc();
  if (avPacket == nullptr) {
    LOG(ERROR) << "uuid=" << params_.loggingUuid
               << " decoder as not able to allocate the packet.";
    return AVERROR_BUFFER_TOO_SMALL;
  }
  avPacket->data = nullptr;
  avPacket->size = 0;

  auto end = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(workingTimeInMs);
  // return true if elapsed time less than timeout
  auto watcher = [end]() -> bool {
    return std::chrono::steady_clock::now() <= end;
  };

  int result = 0;
  size_t decodingErrors = 0;
  bool decodedFrame = false;
  while (!interrupted_ && inRange_.any() && !decodedFrame) {
    if (watcher() == false) {
      LOG(ERROR) << "uuid=" << params_.loggingUuid << " hit ETIMEDOUT";
      result = ETIMEDOUT;
      break;
    }
    result = av_read_frame(inputCtx_, avPacket);
    if (result == AVERROR(EAGAIN)) {
      VLOG(4) << "Decoder is busy...";
      std::this_thread::yield();
      result = 0; // reset error, EAGAIN is not an error at all
      // reset the packet to default settings
      av_packet_unref(avPacket);
      continue;
    } else if (result == AVERROR_EOF) {
      flushStreams();
      VLOG(1) << "End of stream";
      result = ENODATA;
      break;
    } else if (
        result == AVERROR(EPERM) && params_.skipOperationNotPermittedPackets) {
      // reset error, lets skip packets with EPERM
      result = 0;
      // reset the packet to default settings
      av_packet_unref(avPacket);
      continue;
    } else if (result < 0) {
      flushStreams();
      LOG(ERROR) << "uuid=" << params_.loggingUuid
                 << " error detected: " << Util::generateErrorDesc(result);
      break;
    }

    // get stream; if stream cannot be found reset the packet to
    // default settings
    auto stream = findByIndex(avPacket->stream_index);
    if (stream == nullptr || !inRange_.test(stream->getIndex())) {
      av_packet_unref(avPacket);
      continue;
    }

    size_t numConsecutiveNoBytes = 0;
    // it can be only partial decoding of the package bytes
    do {
      // decode package
      bool gotFrame = false;
      bool hasMsg = false;
      // packet either got consumed completely or not at all
      if ((result = processPacket(
               stream, avPacket, &gotFrame, &hasMsg, params_.fastSeek)) < 0) {
        LOG(ERROR) << "uuid=" << params_.loggingUuid
                   << " processPacket failed with code: " << result;
        break;
      }

      if (!gotFrame && params_.maxProcessNoBytes != 0 &&
          ++numConsecutiveNoBytes > params_.maxProcessNoBytes) {
        LOG(ERROR) << "uuid=" << params_.loggingUuid
                   << " exceeding max amount of consecutive no bytes";
        break;
      }
      if (result > 0) {
        numConsecutiveNoBytes = 0;
      }

      decodedFrame |= hasMsg;
    } while (result == 0);

    // post loop check
    if (result < 0) {
      if (params_.maxPackageErrors != 0 && // check errors
          ++decodingErrors >= params_.maxPackageErrors) { // reached the limit
        LOG(ERROR) << "uuid=" << params_.loggingUuid
                   << " exceeding max amount of consecutive package errors";
        break;
      }
    } else {
      decodingErrors = 0; // reset on success
    }

    result = 0;

    av_packet_unref(avPacket);

    if (params_.uniformSampling > 1) {
      if (doSeek_) {
        double duration =
            videoDurationMs_ > 0 ? videoDurationMs_ : params_.expectedDuration;
        double step =
            (duration * AV_TIME_BASE) / (1000 * (params_.uniformSampling - 1));
        avformat_seek_file(
            inputCtx_,
            -1,
            static_cast<int64_t>(step * kFramesDecoded_) + 1,
            static_cast<int64_t>(step * (kFramesDecoded_ + 1)),
            static_cast<int64_t>(step * (kFramesDecoded_ + 1)),
            0);
        ++kFramesDecoded_;
        doSeek_ = false;
      }
    }
  }

  av_packet_free(&avPacket);
  VLOG(2) << "Interrupted loop" << ", interrupted_ " << interrupted_
          << ", inRange_.any() " << inRange_.any() << ", decodedFrame "
          << decodedFrame << ", result " << result;

  // loop can be terminated, either by:
  // 1. explicitly interrupted
  // 3. unrecoverable error or ENODATA (end of stream) or ETIMEDOUT (timeout)
  // 4. decoded frames pts are out of the specified range
  // 5. success decoded frame
  if (interrupted_) {
    return EINTR;
  }
  if (result != 0) {
    return result;
  }
  if (inRange_.none()) {
    return ENODATA;
  }
  return 0;
}

// find stream by stream index
Stream* Decoder::findByIndex(int streamIndex) const {
  auto it = streams_.find(streamIndex);
  return it != streams_.end() ? it->second.get() : nullptr;
}

// find stream by type; note finds only the first stream of a given type
Stream* Decoder::findByType(const MediaFormat& format) const {
  for (auto& stream : streams_) {
    if (stream.second->getMediaFormat().type == format.type) {
      return stream.second.get();
    }
  }
  return nullptr;
}

// given the stream and packet, decode the frame buffers into the
// DecoderOutputMessage data structure via stream::decodePacket function.
int Decoder::processPacket(
    Stream* stream,
    AVPacket* packet,
    bool* gotFrame,
    bool* hasMsg,
    bool fastSeek) {
  // decode package
  int result;
  DecoderOutputMessage msg;
  msg.payload = params_.headerOnly ? nullptr : createByteStorage(0);
  *hasMsg = false;
  if ((result = stream->decodePacket(
           packet, &msg, params_.headerOnly, gotFrame)) >= 0 &&
      *gotFrame) {
    // check end offset
    bool endInRange =
        params_.endOffset <= 0 || msg.header.pts <= params_.endOffset;
    inRange_.set(stream->getIndex(), endInRange);
    // if fastseek is enabled, we're returning the first
    // frame that we decode after (potential) seek.
    // By default, we perform accurate seek to the closest
    // following frame
    bool startCondition = true;
    if (!fastSeek) {
      startCondition = msg.header.pts >= params_.startOffset;
    }
    if (endInRange && startCondition) {
      *hasMsg = pushMsg(std::move(msg));
    }
  }
  return result;
}

bool Decoder::pushMsg(DecoderOutputMessage&& msg) {
  pastDecodedPTS_ = currentDecodedPTS_;
  currentDecodedPTS_ = msg.header.pts;

  if (params_.uniformSampling <= 1) {
    push(std::move(msg));
    return true;
  }

  double duration =
      videoDurationMs_ > 0 ? videoDurationMs_ : params_.expectedDuration;
  double step =
      (duration * AV_TIME_BASE) / (1000 * (params_.uniformSampling - 1));
  if (pastDecodedPTS_ < step * kFramesDecoded_ &&
      step * kFramesDecoded_ <= currentDecodedPTS_) {
    push(std::move(msg));
    doSeek_ = true;
    return true;
  }

  return false;
}

void Decoder::flushStreams() {
  VLOG(1) << "Flushing streams...";
  for (auto& stream : streams_) {
    DecoderOutputMessage msg;
    while (msg.payload = (params_.headerOnly ? nullptr : createByteStorage(0)),
           stream.second->flush(&msg, params_.headerOnly) > 0) {
      // check end offset
      bool endInRange =
          params_.endOffset <= 0 || msg.header.pts <= params_.endOffset;
      inRange_.set(stream.second->getIndex(), endInRange);
      if (endInRange && msg.header.pts >= params_.startOffset) {
        pushMsg(std::move(msg));
      } else {
        msg.payload.reset();
      }
    }
  }
}

int Decoder::decode_all(const DecoderOutCallback& callback) {
  int result;
  do {
    DecoderOutputMessage out;
    if (0 == (result = decode(&out, params_.timeoutMs))) {
      callback(std::move(out));
    }
  } while (result == 0);
  return result;
}
} // namespace ffmpeg
