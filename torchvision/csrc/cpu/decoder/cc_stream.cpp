#include "cc_stream.h"

namespace ffmpeg {

CCStream::CCStream(
    AVFormatContext* inputCtx,
    int index,
    bool convertPtsToWallTime,
    const SubtitleFormat& format)
    : SubtitleStream(inputCtx, index, convertPtsToWallTime, format) {
  format_.type = TYPE_CC;
}

AVCodec* CCStream::findCodec(AVCodecParameters* params) {
  if (params->codec_id == AV_CODEC_ID_BIN_DATA &&
      params->codec_type == AVMEDIA_TYPE_DATA) {
    // obtain subtitles codec
    params->codec_id = AV_CODEC_ID_MOV_TEXT;
    params->codec_type = AVMEDIA_TYPE_SUBTITLE;
  }
  return Stream::findCodec(params);
}

} // namespace ffmpeg
