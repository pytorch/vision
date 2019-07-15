#ifndef DATASETSIMPL_H
#define DATASETSIMPL_H

#include <torch/torch.h>
#include <filesystem>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

namespace vision {
namespace datasets {
namespace datasetsimpl {

inline std::vector<std::string> lsdir(const std::string& path) {
  std::vector<std::string> list;
  for (const auto& ent : std::filesystem::directory_iterator(path))
    list.push_back(ent.path().filename());
  return list;
}

inline std::string tolower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  return str;
}

inline void sort_names(std::vector<std::string>& data) {
  auto comp = [](const std::string& A, const std::string& B) {
    return tolower(A) < tolower(B);
  };
  std::sort(data.begin(), data.end(), comp);
}

inline bool isdir(const std::string& path) {
  return std::filesystem::is_directory(path);
}

inline bool isfile(const std::string& path) {
  return std::filesystem::is_regular_file(path);
}

inline bool exists(const std::string& path) {
  return std::filesystem::exists(path);
}

inline bool mkpath(const std::string& path) {
  return std::filesystem::create_directories(path);
}

inline std::string join(const std::string& str) {
  return str;
}

template <typename... Tail>
inline std::string join(const std::string& head, Tail&&... tail) {
  return std::filesystem::path(head).append(join(tail...)).string();
}

} // namespace datasetsimpl
} // namespace datasets
} // namespace vision

#endif // DATASETSIMPL_H
