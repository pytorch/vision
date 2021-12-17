extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}

inline bool check(int ret, int line)
{
  if (ret < 0) {
    printf("Error %d at line %d in demuxer.h.\n", ret, line);
    return false;
  }
  return true;
}

#define check_for_errors(call) check(call, __LINE__)

class Demuxer {
  private:
    AVFormatContext *fmtCtx = NULL;
    AVBSFContext *bsfCtx = NULL;
    AVPacket pkt, pktFiltered;
    AVCodecID eVideoCodec;
    uint8_t *pDataWithHeader = NULL;
    bool bMp4H264, bMp4HEVC, bMp4MPEG4;
    unsigned int frameCount = 0;
    int iVideoStream;
    int64_t userTimeScale = 0;
    double timeBase = 0.0;

  public:
    Demuxer(const char *filePath, int64_t timeScale = 1000 /*Hz*/)
    {
        avformat_network_init();
        check_for_errors(avformat_open_input(&fmtCtx, filePath, NULL, NULL));
        if (!fmtCtx) {
            printf("No AVFormatContext provided.\n");
            return;
        }

        check_for_errors(avformat_find_stream_info(fmtCtx, NULL));
        iVideoStream = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (iVideoStream < 0) {
            printf("FFmpeg error: %d, could not find stream in input file\n", __LINE__);
            return;
        }

        eVideoCodec = fmtCtx->streams[iVideoStream]->codecpar->codec_id;
        AVRational rTimeBase = fmtCtx->streams[iVideoStream]->time_base;
        timeBase = av_q2d(rTimeBase);
        userTimeScale = timeScale;

        bMp4H264 = eVideoCodec == AV_CODEC_ID_H264 && (
                !strcmp(fmtCtx->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtCtx->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtCtx->iformat->long_name, "Matroska / WebM"));
        bMp4HEVC = eVideoCodec == AV_CODEC_ID_HEVC && (
                !strcmp(fmtCtx->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtCtx->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtCtx->iformat->long_name, "Matroska / WebM"));
        bMp4MPEG4 = eVideoCodec == AV_CODEC_ID_MPEG4 && (
                !strcmp(fmtCtx->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtCtx->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtCtx->iformat->long_name, "Matroska / WebM"));

        av_init_packet(&pkt);
        pkt.data = NULL;
        pkt.size = 0;
        av_init_packet(&pktFiltered);
        pktFiltered.data = NULL;
        pktFiltered.size = 0;

        if (bMp4H264) {
            const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf) {
                printf("FFmpeg error: %d, av_bsf_get_by_name() failed\n", __LINE__);
                return;
            }
            check_for_errors(av_bsf_alloc(bsf, &bsfCtx));
            avcodec_parameters_copy(bsfCtx->par_in, fmtCtx->streams[iVideoStream]->codecpar);
            check_for_errors(av_bsf_init(bsfCtx));
        }
        if (bMp4HEVC) {
            const AVBitStreamFilter *bsf = av_bsf_get_by_name("hevc_mp4toannexb");
            if (!bsf) {
                printf("FFmpeg error: %d, av_bsf_get_by_name() failed\n", __LINE__);
                return;
            }
            check_for_errors(av_bsf_alloc(bsf, &bsfCtx));
            avcodec_parameters_copy(bsfCtx->par_in, fmtCtx->streams[iVideoStream]->codecpar);
            check_for_errors(av_bsf_init(bsfCtx));
        }
    }
    ~Demuxer()
    {
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
        if (pDataWithHeader) {
            av_free(pDataWithHeader);
        }
    }

    AVCodecID GetVideoCodec()
    {
        return eVideoCodec;
    }

    bool Demux(uint8_t **video, unsigned long *videoBytes)
    {
        if (!fmtCtx) {
            return false;
        }
        *videoBytes = 0;

        if (pkt.data) {
            av_packet_unref(&pkt);
        }
        int e = 0;
        while ((e = av_read_frame(fmtCtx, &pkt)) >= 0 && pkt.stream_index != iVideoStream) {
            av_packet_unref(&pkt);
        }
        if (e < 0) {
            return false;
        }

        if (bMp4H264 || bMp4HEVC) {
            if (pktFiltered.data) {
                av_packet_unref(&pktFiltered);
            }
            check_for_errors(av_bsf_send_packet(bsfCtx, &pkt));
            check_for_errors(av_bsf_receive_packet(bsfCtx, &pktFiltered));
            *video = pktFiltered.data;
            *videoBytes = pktFiltered.size;
        } else {
            if (bMp4MPEG4 && (frameCount == 0)) {
                int extraDataSize = fmtCtx->streams[iVideoStream]->codecpar->extradata_size;

                if (extraDataSize > 0) {
                    pDataWithHeader = (uint8_t *)av_malloc(extraDataSize + pkt.size - 3 * sizeof(uint8_t));
                    if (!pDataWithHeader) {
                        printf("FFmpeg error: %d\n",  __LINE__);
                        return false;
                    }
                    memcpy(pDataWithHeader, fmtCtx->streams[iVideoStream]->codecpar->extradata, extraDataSize);
                    memcpy(pDataWithHeader+extraDataSize, pkt.data+3, pkt.size - 3 * sizeof(uint8_t));
                    *video = pDataWithHeader;
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
};

inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id)
{
    switch (id) {
      case AV_CODEC_ID_MPEG1VIDEO : return cudaVideoCodec_MPEG1;
      case AV_CODEC_ID_MPEG2VIDEO : return cudaVideoCodec_MPEG2;
      case AV_CODEC_ID_MPEG4      : return cudaVideoCodec_MPEG4;
      case AV_CODEC_ID_WMV3       :
      case AV_CODEC_ID_VC1        : return cudaVideoCodec_VC1;
      case AV_CODEC_ID_H264       : return cudaVideoCodec_H264;
      case AV_CODEC_ID_HEVC       : return cudaVideoCodec_HEVC;
      case AV_CODEC_ID_VP8        : return cudaVideoCodec_VP8;
      case AV_CODEC_ID_VP9        : return cudaVideoCodec_VP9;
      case AV_CODEC_ID_MJPEG      : return cudaVideoCodec_JPEG;
      case AV_CODEC_ID_AV1        : return cudaVideoCodec_AV1;
      default                     : return cudaVideoCodec_NumCodecs;
    }
}
