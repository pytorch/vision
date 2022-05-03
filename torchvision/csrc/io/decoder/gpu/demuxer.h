extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}

class Demuxer {
 private:
  AVFormatContext* fmtCtx = NULL;
  AVBSFContext* bsfCtx = NULL;
  AVPacket pkt, pktFiltered;
  AVCodecID eVideoCodec;
  uint8_t* dataWithHeader = NULL;
  bool bMp4H264, bMp4HEVC, bMp4MPEG4;
  unsigned int frameCount = 0;
  int iVideoStream;
  double timeBase = 0.0;

 public:
  Demuxer(const char* filePath) {
    avformat_network_init();
    TORCH_CHECK(
        0 <= avformat_open_input(&fmtCtx, filePath, NULL, NULL),
        "avformat_open_input() failed at line ",
        __LINE__,
        " in demuxer.h\n");
    if (!fmtCtx) {
      TORCH_CHECK(
          false,
          "Encountered NULL AVFormatContext at line ",
          __LINE__,
          " in demuxer.h\n");
    }

    TORCH_CHECK(
        0 <= avformat_find_stream_info(fmtCtx, NULL),
        "avformat_find_stream_info() failed at line ",
        __LINE__,
        " in demuxer.h\n");
    iVideoStream =
        av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (iVideoStream < 0) {
      TORCH_CHECK(
          false,
          "av_find_best_stream() failed at line ",
          __LINE__,
          " in demuxer.h\n");
    }

    eVideoCodec = fmtCtx->streams[iVideoStream]->codecpar->codec_id;
    AVRational rTimeBase = fmtCtx->streams[iVideoStream]->time_base;
    timeBase = av_q2d(rTimeBase);

    bMp4H264 = eVideoCodec == AV_CODEC_ID_H264 &&
        (!strcmp(fmtCtx->iformat->long_name, "QuickTime / MOV") ||
         !strcmp(fmtCtx->iformat->long_name, "FLV (Flash Video)") ||
         !strcmp(fmtCtx->iformat->long_name, "Matroska / WebM"));
    bMp4HEVC = eVideoCodec == AV_CODEC_ID_HEVC &&
        (!strcmp(fmtCtx->iformat->long_name, "QuickTime / MOV") ||
         !strcmp(fmtCtx->iformat->long_name, "FLV (Flash Video)") ||
         !strcmp(fmtCtx->iformat->long_name, "Matroska / WebM"));
    bMp4MPEG4 = eVideoCodec == AV_CODEC_ID_MPEG4 &&
        (!strcmp(fmtCtx->iformat->long_name, "QuickTime / MOV") ||
         !strcmp(fmtCtx->iformat->long_name, "FLV (Flash Video)") ||
         !strcmp(fmtCtx->iformat->long_name, "Matroska / WebM"));

    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;
    av_init_packet(&pktFiltered);
    pktFiltered.data = NULL;
    pktFiltered.size = 0;

    if (bMp4H264) {
      const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
      if (!bsf) {
        TORCH_CHECK(
            false,
            "av_bsf_get_by_name() failed at line ",
            __LINE__,
            " in demuxer.h\n");
      }
      TORCH_CHECK(
          0 <= av_bsf_alloc(bsf, &bsfCtx),
          "av_bsf_alloc() failed at line ",
          __LINE__,
          " in demuxer.h\n");
      avcodec_parameters_copy(
          bsfCtx->par_in, fmtCtx->streams[iVideoStream]->codecpar);
      TORCH_CHECK(
          0 <= av_bsf_init(bsfCtx),
          "av_bsf_init() failed at line ",
          __LINE__,
          " in demuxer.h\n");
    }
    if (bMp4HEVC) {
      const AVBitStreamFilter* bsf = av_bsf_get_by_name("hevc_mp4toannexb");
      if (!bsf) {
        TORCH_CHECK(
            false,
            "av_bsf_get_by_name() failed at line ",
            __LINE__,
            " in demuxer.h\n");
      }
      TORCH_CHECK(
          0 <= av_bsf_alloc(bsf, &bsfCtx),
          "av_bsf_alloc() failed at line ",
          __LINE__,
          " in demuxer.h\n");
      avcodec_parameters_copy(
          bsfCtx->par_in, fmtCtx->streams[iVideoStream]->codecpar);
      TORCH_CHECK(
          0 <= av_bsf_init(bsfCtx),
          "av_bsf_init() failed at line ",
          __LINE__,
          " in demuxer.h\n");
    }
  }

  ~Demuxer() {
    if (!fmtCtx) {
      return;
    }
    if (pkt.data) {
      av_packet_unref(&pkt);
    }
    if (pktFiltered.data) {
      av_packet_unref(&pktFiltered);
    }
    if (bsfCtx) {
      av_bsf_free(&bsfCtx);
    }
    avformat_close_input(&fmtCtx);
    if (dataWithHeader) {
      av_free(dataWithHeader);
    }
  }

  AVCodecID get_video_codec() {
    return eVideoCodec;
  }

  double get_duration() const {
    return (double)fmtCtx->duration / AV_TIME_BASE;
  }

  double get_fps() const {
    return av_q2d(fmtCtx->streams[iVideoStream]->r_frame_rate);
  }

  bool demux(uint8_t** video, unsigned long* videoBytes) {
    if (!fmtCtx) {
      return false;
    }
    *videoBytes = 0;

    if (pkt.data) {
      av_packet_unref(&pkt);
    }
    int e = 0;
    while ((e = av_read_frame(fmtCtx, &pkt)) >= 0 &&
           pkt.stream_index != iVideoStream) {
      av_packet_unref(&pkt);
    }
    if (e < 0) {
      return false;
    }

    if (bMp4H264 || bMp4HEVC) {
      if (pktFiltered.data) {
        av_packet_unref(&pktFiltered);
      }
      TORCH_CHECK(
          0 <= av_bsf_send_packet(bsfCtx, &pkt),
          "av_bsf_send_packet() failed at line ",
          __LINE__,
          " in demuxer.h\n");
      TORCH_CHECK(
          0 <= av_bsf_receive_packet(bsfCtx, &pktFiltered),
          "av_bsf_receive_packet() failed at line ",
          __LINE__,
          " in demuxer.h\n");
      *video = pktFiltered.data;
      *videoBytes = pktFiltered.size;
    } else {
      if (bMp4MPEG4 && (frameCount == 0)) {
        int extraDataSize =
            fmtCtx->streams[iVideoStream]->codecpar->extradata_size;

        if (extraDataSize > 0) {
          dataWithHeader = (uint8_t*)av_malloc(
              extraDataSize + pkt.size - 3 * sizeof(uint8_t));
          if (!dataWithHeader) {
            TORCH_CHECK(
                false,
                "av_malloc() failed at line ",
                __LINE__,
                " in demuxer.h\n");
          }
          memcpy(
              dataWithHeader,
              fmtCtx->streams[iVideoStream]->codecpar->extradata,
              extraDataSize);
          memcpy(
              dataWithHeader + extraDataSize,
              pkt.data + 3,
              pkt.size - 3 * sizeof(uint8_t));
          *video = dataWithHeader;
          *videoBytes = extraDataSize + pkt.size - 3 * sizeof(uint8_t);
        }
      } else {
        *video = pkt.data;
        *videoBytes = pkt.size;
      }
    }
    frameCount++;
    return true;
  }

  void seek(double timestamp, int flag) {
    int64_t time = timestamp * AV_TIME_BASE;
    TORCH_CHECK(
        0 <= av_seek_frame(fmtCtx, -1, time, flag),
        "av_seek_frame() failed at line ",
        __LINE__,
        " in demuxer.h\n");
  }
};

inline cudaVideoCodec ffmpeg_to_codec(AVCodecID id) {
  switch (id) {
    case AV_CODEC_ID_MPEG1VIDEO:
      return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO:
      return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4:
      return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_WMV3:
    case AV_CODEC_ID_VC1:
      return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264:
      return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC:
      return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8:
      return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9:
      return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG:
      return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_AV1:
      return cudaVideoCodec_AV1;
    default:
      return cudaVideoCodec_NumCodecs;
  }
}
