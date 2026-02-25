#include "read_write_file.h"

#include <sys/stat.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace vision {
namespace image {

using namespace vision::stable;

#ifdef _WIN32
namespace {
std::wstring utf8_decode(const std::string& str) {
  if (str.empty()) {
    return std::wstring();
  }
  int size_needed = MultiByteToWideChar(
      CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
  VISION_CHECK(size_needed > 0, "Error converting the content to Unicode");
  std::wstring wstrTo(size_needed, 0);
  MultiByteToWideChar(
      CP_UTF8,
      0,
      str.c_str(),
      static_cast<int>(str.size()),
      &wstrTo[0],
      size_needed);
  return wstrTo;
}
} // namespace
#endif

Tensor read_file(std::string filename) {
#ifdef _WIN32
  // According to
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/stat-functions?view=vs-2019,
  // we should use struct __stat64 and _wstat64 for 64-bit file size on Windows.
  struct __stat64 stat_buf;
  auto fileW = utf8_decode(filename);
  int rc = _wstat64(fileW.c_str(), &stat_buf);
#else
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
#endif
  // errno is a variable defined in errno.h
  VISION_CHECK(
      rc == 0, "[Errno ", errno, "] ", strerror(errno), ": '", filename, "'");

  int64_t size = stat_buf.st_size;

  VISION_CHECK(size > 0, "Expected a non empty file");

#ifdef _WIN32
  // On Windows, read file manually since torch::from_file may not handle UTF-8
  FILE* infile = _wfopen(fileW.c_str(), L"rb");
#else
  // Use fopen/fread instead of from_file for stable ABI compatibility
  FILE* infile = fopen(filename.c_str(), "rb");
#endif

  VISION_CHECK(infile != nullptr, "Error opening input file");

  auto data = emptyCPU({size}, kByte);
  auto dataBytes = data.mutable_data_ptr<uint8_t>();

  fread(dataBytes, sizeof(uint8_t), size, infile);
  fclose(infile);

  return data;
}

void write_file(std::string filename, const Tensor& data) {
  // Check that the input tensor is on CPU
  VISION_CHECK(data.device().type() == kCPU, "Input tensor should be on CPU");

  // Check that the input tensor dtype is uint8
  VISION_CHECK(
      data.scalar_type() == kByte, "Input tensor dtype should be uint8");

  // Check that the input tensor is 3-dimensional
  VISION_CHECK(data.dim() == 1, "Input data should be a 1-dimensional tensor");

  auto fileBytes = data.const_data_ptr<uint8_t>();
  auto fileCStr = filename.c_str();
#ifdef _WIN32
  auto fileW = utf8_decode(filename);
  FILE* outfile = _wfopen(fileW.c_str(), L"wb");
#else
  FILE* outfile = fopen(fileCStr, "wb");
#endif

  VISION_CHECK(outfile != nullptr, "Error opening output file");

  fwrite(fileBytes, sizeof(uint8_t), data.numel(), outfile);
  fclose(outfile);
}

} // namespace image
} // namespace vision
