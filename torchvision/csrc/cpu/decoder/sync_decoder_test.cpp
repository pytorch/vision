// Copyright 2004-present Facebook. All Rights Reserved.

#include <c10/util/Logging.h>
#include <gtest/gtest.h>
#include "sync_decoder.h"

using namespace ffmpeg;

TEST(SyncDecoder, Test) {
  SyncDecoder decoder;
  DecoderParameters params;
  params.timeoutMs = 10000;
  params.startOffsetMs = 1000;
  params.formats = {MediaFormat(), MediaFormat(0), MediaFormat('0')};
  params.uri = "pytorch/vision/test/assets/videos/R6llTwEh07w.mp4";
  CHECK(decoder.init(params, nullptr));
  DecoderOutputMessage out;
  while (0 == decoder.decode(&out, 100)) {
    LOG(INFO) << "Decoded frame, timestamp(us): " << out.header.pts;
  }
  decoder.shutdown();
}
