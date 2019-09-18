#include "util.h"

using namespace std;

namespace util {

unique_ptr<DecoderParameters> getDecoderParams(
    double seekFrameMargin,
    int64_t getPtsOnly,
    int64_t readVideoStream,
    int videoWidth,
    int videoHeight,
    int videoMinDimension,
    int64_t videoStartPts,
    int64_t videoEndPts,
    int videoTimeBaseNum,
    int videoTimeBaseDen,
    int64_t readAudioStream,
    int audioSamples,
    int audioChannels,
    int64_t audioStartPts,
    int64_t audioEndPts,
    int audioTimeBaseNum,
    int audioTimeBaseDen) {
  unique_ptr<DecoderParameters> params = make_unique<DecoderParameters>();

  if (readVideoStream == 1) {
    params->formats.emplace(TYPE_VIDEO, MediaFormat(TYPE_VIDEO));
    MediaFormat& videoFormat = params->formats[TYPE_VIDEO];

    videoFormat.format.video.width = videoWidth;
    videoFormat.format.video.height = videoHeight;
    videoFormat.format.video.minDimension = videoMinDimension;
    videoFormat.format.video.startPts = videoStartPts;
    videoFormat.format.video.endPts = videoEndPts;
    videoFormat.format.video.timeBaseNum = videoTimeBaseNum;
    videoFormat.format.video.timeBaseDen = videoTimeBaseDen;
  }

  if (readAudioStream == 1) {
    params->formats.emplace(TYPE_AUDIO, MediaFormat(TYPE_AUDIO));
    MediaFormat& audioFormat = params->formats[TYPE_AUDIO];

    audioFormat.format.audio.samples = audioSamples;
    audioFormat.format.audio.channels = audioChannels;
    audioFormat.format.audio.startPts = audioStartPts;
    audioFormat.format.audio.endPts = audioEndPts;
    audioFormat.format.audio.timeBaseNum = audioTimeBaseNum;
    audioFormat.format.audio.timeBaseDen = audioTimeBaseDen;
  }

  params->seekFrameMargin = seekFrameMargin;
  params->getPtsOnly = getPtsOnly;

  return params;
}

} // namespace util
