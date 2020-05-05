#include <c10/util/Logging.h>
#include <dirent.h>
#include <gtest/gtest.h>
#include "util.h"

TEST(Util, TestSetFormatDimensions) {
  // clang-format off
  const size_t test_cases[][9] = {
      // (userW, userH, srcW, srcH, minDimension, maxDimension, cropImage, destW, destH)
      {0, 0, 172, 128, 0, 0, 0, 172, 128},    // #1
      {86, 0, 172, 128, 0, 0, 0, 86, 64},     // #2
      {64, 0, 128, 172, 0, 0, 0, 64, 86},     // #2
      {0, 32, 172, 128, 0, 0, 0, 43, 32},     // #3
      {32, 0, 128, 172, 0, 0, 0, 32, 43},     // #3
      {60, 50, 172, 128, 0, 0, 0, 60, 50},    // #4
      {50, 60, 128, 172, 0, 0, 0, 50, 60},    // #4
      {86, 40, 172, 128, 0, 0, 1, 86, 64},    // #5
      {86, 92, 172, 128, 0, 0, 1, 124, 92},   // #5
      {0, 0, 172, 128, 256, 0, 0, 344, 256},  // #6
      {0, 0, 128, 172, 256, 0, 0, 256, 344},  // #6
      {0, 0, 128, 172, 0, 344, 0, 256, 344},  // #7
      {0, 0, 172, 128, 0, 344, 0, 344, 256},  // #7
      {0, 0, 172, 128, 100, 344, 0, 344, 100},// #8
      {0, 0, 128, 172, 100, 344, 0, 100, 344} // #8
  };
  // clang-format onn

  for (const auto& tc : test_cases) {
      size_t destW = 0;
      size_t destH = 0;
      ffmpeg::Util::setFormatDimensions(destW, destH, tc[0], tc[1], tc[2], tc[3], tc[4], tc[5], tc[6]);
      CHECK(destW == tc[7]);
      CHECK(destH == tc[8]);
  }
}
