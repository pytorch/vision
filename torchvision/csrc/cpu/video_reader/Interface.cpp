#include "Interface.h"

void DecoderOutput::initMediaType(MediaType mediaType, FormatUnion format) {
  MediaData mediaData(format);
  media_data_.emplace(mediaType, std::move(mediaData));
}

void DecoderOutput::addMediaFrame(
    MediaType mediaType,
    std::unique_ptr<DecodedFrame> frame) {
  if (media_data_.find(mediaType) != media_data_.end()) {
    VLOG(1) << "media type: " << mediaType
            << " add frame with pts: " << frame->pts_;
    media_data_[mediaType].frames_.push_back(std::move(frame));
  } else {
    VLOG(1) << "media type: " << mediaType << " not found. Skip the frame.";
  }
}

void DecoderOutput::clear() {
  media_data_.clear();
}
