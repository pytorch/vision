#include "read_write_file_cpu.h"

// According to
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/stat-functions?view=vs-2019,
// we should use _stat64 for 64-bit file size on Windows.
#ifdef _WIN32
#ifdef UNICODE
#define VISION_STAT _wstat64
#else
#define VISION_STAT _stat64
#endif
#else
#define VISION_STAT stat
#endif

torch::Tensor read_file(VISION_STRING filename) {
  struct VISION_STAT stat_buf;
  int rc = VISION_STAT(filename.c_str(), &stat_buf);
  // errno is a variable defined in errno.h
#if defined(_WIN32) && defined(UNICODE)
  size_t size = wcslen(filename) + 1;
  std::unique_ptr<char> filename_ascii_raw(new char[size]);
  std::wcstombs(filename_ascii_raw.get(), filename, size);
  std::string filename_ascii(filename_ascii_raw.get());
#else
  std::string& filename_ascii = filename;  
#endif

  TORCH_CHECK(
      rc == 0, "[Errno ", errno, "] ", strerror(errno), ": '", filename_ascii, "'");

  int64_t size = stat_buf.st_size;

  TORCH_CHECK(size > 0, "Expected a non empty file");

#ifdef _WIN32
  // Expected error for Unicode variant here
  auto data =
      torch::from_file(filename_ascii, /*shared=*/false, /*size=*/size, torch::kU8)
          .clone();
#else
  auto data =
      torch::from_file(filename_ascii, /*shared=*/false, /*size=*/size, torch::kU8);
#endif

  return data;
}

void write_file(VISION_STRING filename, torch::Tensor& data) {
  // Check that the input tensor is on CPU
  TORCH_CHECK(data.device() == torch::kCPU, "Input tensor should be on CPU");

  // Check that the input tensor dtype is uint8
  TORCH_CHECK(data.dtype() == torch::kU8, "Input tensor dtype should be uint8");

  // Check that the input tensor is 3-dimensional
  TORCH_CHECK(data.dim() == 1, "Input data should be a 1-dimensional tensor");

  auto fileBytes = data.data_ptr<uint8_t>();
  auto fileCStr = filename.c_str();
#if defined(_WIN32) && defined(UNICODE)
  FILE* outfile = _wfopen(fileCStr, L"wb");
#else
  FILE* outfile = fopen(fileCStr, "wb");
#endif

  TORCH_CHECK(outfile != nullptr, "Error opening output file");

  fwrite(fileBytes, sizeof(uint8_t), data.numel(), outfile);
  fclose(outfile);
}
