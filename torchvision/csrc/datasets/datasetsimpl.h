#ifndef DATASETSIMPL_H
#define DATASETSIMPL_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <filesystem>

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

namespace vision {
namespace datasets {
namespace datasetsimpl {

std::vector<std::string> lsdir(const std::string& path);

std::string tolower(std::string str);

void sort_names(std::vector<std::string>& data);

bool isdir(const std::string& path);

bool isfile(const std::string& path);

bool exists(const std::string& path);

bool mkpath(const std::string& path);

std::string absolute_path(const std::string& path);

inline std::string join(const std::string& str) {
  return str;
}
template <typename... Tail>
inline std::string join(const std::string& head, Tail&&... tail) {
  return std::filesystem::path(head).append(join(tail...)).string();
}

torch::Tensor read_image(
    const std::string& path,
    std::function<cv::Mat(const cv::Mat&)> transform);

std::function<cv::Mat(const cv::Mat&)> make_transform(
    int width,
    int height,
    cv::ColorConversionCodes code);

inline auto rgb_transform = make_transform(224, 224, cv::COLOR_BGR2RGB);
inline auto gray_transform = make_transform(224, 224, cv::COLOR_BGR2GRAY);

} // namespace datasetsimpl
} // namespace datasets
} // namespace vision

#endif // DATASETSIMPL_H
