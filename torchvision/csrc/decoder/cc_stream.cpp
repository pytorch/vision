// Copyright 2004-present Facebook. All Rights Reserved.

#include "cc_stream.h"

namespace ffmpeg {

AVCodec* CCStream::findCodec(AVCodecContext* ctx) {
  if (ctx->codec_id == AV_CODEC_ID_BIN_DATA &&
      ctx->codec_type == AVMEDIA_TYPE_DATA) {
    // obtain subtitles codec
    ctx->codec_id = AV_CODEC_ID_MOV_TEXT;
    ctx->codec_type = AVMEDIA_TYPE_SUBTITLE;
  }
  return Stream::findCodec(ctx);
}

void CCStream::setHeader(DecoderHeader* header) {
  SubtitleStream::setHeader(header);
  header->format.type = TYPE_CC;
}

} // namespace ffmpeg
