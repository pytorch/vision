#include "read_write_file.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include <sys/stat.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace vision {
namespace image {

#ifndef _WIN32
namespace {
// Shim for the from_file op missing in stable ABI.
// TODO(stable-abi): remove once from_file lands in the stable ABI upstream.
torch::stable::Tensor stable_from_file(
    const std::string& filename,
    bool shared,
    int64_t size,
    torch::headeronly::ScalarType dtype) {
  const auto num_args = 7;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(filename),
      torch::stable::detail::from(std::optional<bool>(shared)),
      torch::stable::detail::from(std::optional<int64_t>(size)),
      torch::stable::detail::from(
          std::optional<torch::headeronly::ScalarType>(dtype)),
      torch::stable::detail::from(std::nullopt), // layout
      torch::stable::detail::from(std::nullopt), // device
      torch::stable::detail::from(std::nullopt)}; // pin_memory
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::from_file", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}
} // namespace
#endif

#ifdef _WIN32
namespace {
std::wstring utf8_decode(const std::string& str) {
  if (str.empty()) {
    return std::wstring();
  }
  int size_needed = MultiByteToWideChar(
      CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
  STD_TORCH_CHECK(size_needed > 0, "Error converting the content to Unicode");
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

torch::stable::Tensor read_file(const std::string& filename) {
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
  STD_TORCH_CHECK(
      rc == 0, "[Errno ", errno, "] ", strerror(errno), ": '", filename, "'");

  int64_t size = stat_buf.st_size;

  STD_TORCH_CHECK(size > 0, "Expected a non empty file");

#ifdef _WIN32
  // TODO: Once torch::from_file handles UTF-8 paths correctly, we should move
  // back to use the following implementation since it uses file mapping.
  //   auto data =
  //       torch::from_file(filename, /*shared=*/false, /*size=*/size,
  //       torch::kU8).clone()
  FILE* infile = _wfopen(fileW.c_str(), L"rb");

  STD_TORCH_CHECK(infile != nullptr, "Error opening input file");

  auto data = torch::stable::empty({size}, torch::headeronly::ScalarType::Byte);
  auto dataBytes = data.mutable_data_ptr<uint8_t>();

  fread(dataBytes, sizeof(uint8_t), size, infile);
  fclose(infile);
#else
  auto data = stable_from_file(
      filename,
      /*shared=*/false,
      /*size=*/size,
      torch::headeronly::ScalarType::Byte);
#endif

  return data;
}

torch::stable::Tensor write_file(
    const std::string& filename,
    torch::stable::Tensor& data) {
  // Check that the input tensor is on CPU
  STD_TORCH_CHECK(data.is_cpu(), "Input tensor should be on CPU");

  // Check that the input tensor dtype is uint8
  STD_TORCH_CHECK(
      data.scalar_type() == torch::headeronly::ScalarType::Byte,
      "Input tensor dtype should be uint8");

  // Check that the input tensor is 3-dimensional
  STD_TORCH_CHECK(
      data.dim() == 1, "Input data should be a 1-dimensional tensor");

  auto fileBytes = data.const_data_ptr<uint8_t>();
  auto fileCStr = filename.c_str();
#ifdef _WIN32
  auto fileW = utf8_decode(filename);
  FILE* outfile = _wfopen(fileW.c_str(), L"wb");
#else
  FILE* outfile = fopen(fileCStr, "wb");
#endif

  STD_TORCH_CHECK(outfile != nullptr, "Error opening output file");

  fwrite(fileBytes, sizeof(uint8_t), data.numel(), outfile);
  fclose(outfile);

  return data;
}

STABLE_TORCH_LIBRARY_FRAGMENT(image, m) {
  m.def("read_file(str filename) -> Tensor");
  // write_file returns its input so TorchScript DCE keeps the call alive
  // since a stable def cannot express AliasAnalysisKind::CONSERVATIVE:
  // https://github.com/pytorch/pytorch/issues/189309
  m.def("write_file(str filename, Tensor data) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(image, CompositeExplicitAutograd, m) {
  m.impl("read_file", TORCH_BOX(&read_file));
  m.impl("write_file", TORCH_BOX(&write_file));
}

} // namespace image
} // namespace vision
