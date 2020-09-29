#include "read_write_file_cpu.h"

torch::Tensor read_file(std::string filename) {
  // CHECK if this only works on Windows for files smaller than 2GB
  // https://stackoverflow.com/questions/5840148/how-can-i-get-a-files-size-in-c
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  // errno is a variable defined in errno.h
  TORCH_CHECK(
      rc == 0, "[Errno ", errno, "] ", strerror(errno), ": '", filename, "'");
  
  int64_t size = stat_buf.st_size;

  TORCH_CHECK(size > 0, "Expected a non empty file");

  auto data =
    torch::from_file(filename, /*shared=*/false, /*size=*/size, torch::kU8);

  return data;
}
