#include "datasetsimpl.h"

std::vector<std::string> vision::datasets::datasetsimpl::lsdir(
    const std::string& path) {
  std::vector<std::string> list;
  for (const auto& ent : std::filesystem::directory_iterator(path))
    list.push_back(ent.path().filename());
  return list;
}

std::string vision::datasets::datasetsimpl::tolower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  return str;
}

void vision::datasets::datasetsimpl::sort_names(
    std::vector<std::string>& data) {
  auto comp = [](const std::string& A, const std::string& B) {
    return tolower(A) < tolower(B);
  };
  std::sort(data.begin(), data.end(), comp);
}

bool vision::datasets::datasetsimpl::isdir(const std::string& path) {
  return std::filesystem::is_directory(path);
}

bool vision::datasets::datasetsimpl::isfile(const std::string& path) {
  return std::filesystem::is_regular_file(path);
}

bool vision::datasets::datasetsimpl::exists(const std::string& path) {
  return std::filesystem::exists(path);
}

bool vision::datasets::datasetsimpl::mkpath(const std::string& path) {
  if (exists(path))
    return true;
  return std::filesystem::create_directories(path);
}

torch::Tensor vision::datasets::datasetsimpl::read_image(
    const std::string& path,
    std::function<cv::Mat(const cv::Mat&)> transform) {
  auto mat = cv::imread(path);
  TORCH_CHECK(!mat.empty(), "Failed to read image \"", path, "\".");

  // TODO make channels variable

  mat = transform(mat);
  std::vector<torch::Tensor> tensors;
  std::vector<cv::Mat> channels(size_t(mat.channels()));
  cv::split(mat, channels);

  for (auto& channel : channels)
    tensors.push_back(
        torch::from_blob(channel.ptr(), {mat.rows, mat.cols}, torch::kUInt8));

  return torch::cat(tensors)
      .view({mat.channels(), mat.rows, mat.cols})
      .to(torch::kFloat);
}

std::string vision::datasets::datasetsimpl::absolute_path(
    const std::string& path) {
  return std::filesystem::absolute(path).string();
}

std::function<cv::Mat(const cv::Mat&)> vision::datasets::datasetsimpl::
    make_transform(int width, int height, cv::ColorConversionCodes code) {
  return [width, height, code](const cv::Mat& mat) {
    cv::Mat new_mat;
    cv::resize(mat, new_mat, cv::Size(width, height));
    cv::cvtColor(new_mat, new_mat, code);
    return new_mat;
  };
}
