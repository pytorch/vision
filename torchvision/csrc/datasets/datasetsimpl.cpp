#include "datasetsimpl.h"

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace vision {
namespace datasets {
namespace datasetsimpl {
std::vector<std::string> lsdir(const std::string& path) {
  std::vector<std::string> list;
  auto dp = opendir(path.c_str());

  if (dp != nullptr) {
    auto ep = readdir(dp);

    while (ep != nullptr) {
      std::string name = ep->d_name;
      if (name != "." && name != "..")
        list.emplace_back(std::move(name));
    }

    closedir(dp);
  }

  return list;
}

std::string tolower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  return str;
}

inline bool comp(const std::string& A, const std::string& B) {
  return tolower(A) < tolower(B);
};

void sort_names(std::vector<std::string>& data) {
  std::sort(data.begin(), data.end(), comp);
}

bool isdir(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0)
    return st.st_mode & S_IFDIR;
  return false;
}

bool isfile(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0)
    return st.st_mode & S_IFREG;
  return false;
}

bool exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

torch::Tensor read_image(const std::string& path) {
  auto mat = cv::imread(path);
  TORCH_CHECK(!mat.empty(), "Failed to read image \"", path, "\".");

  cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
  std::vector<torch::Tensor> tensors;
  std::vector<cv::Mat> channels(size_t(mat.channels()));
  cv::split(mat, channels);

  for (auto& channel : channels)
    tensors.push_back(
        torch::from_blob(channel.ptr(), {mat.rows, mat.cols}, torch::kUInt8));

  auto output = torch::cat(tensors)
                    .view({mat.channels(), mat.rows, mat.cols})
                    .to(torch::kFloat);
  return output / 255;
}

std::string absolute_path(const std::string& path) {
  char rpath[PATH_MAX];
  realpath(path.c_str(), rpath);
  return std::string(rpath);
}

} // namespace datasetsimpl
} // namespace datasets
} // namespace vision
