#include "caltech.h"

namespace vision {
namespace datasets {

Caltech101::Caltech101(
    std::string root,
    std::function<cv::Mat(const cv::Mat&)> transform)
    : cv_transform(transform) {
  root = datasetsimpl::absolute_path(datasetsimpl::join(root, "caltech101"));
  this->root = root;

  TORCH_CHECK(
      datasetsimpl::mkpath(root),
      "Failed to create directory \"",
      root,
      "\" for Caltech101 dataset");

  TORCH_CHECK(checkIntegrity(), "Caltech101 dataset not found or corrupted.")

  categories =
      datasetsimpl::lsdir(datasetsimpl::join(root, "101_ObjectCategories"));
  datasetsimpl::sort_names(categories);

  auto it =
      std::find(categories.begin(), categories.end(), "BACKGROUND_Google");
  if (it != categories.end())
    categories.erase(it);

  long count = 0;
  for (const auto& category : categories) {
    auto files = datasetsimpl::lsdir(
        datasetsimpl::join(root, "101_ObjectCategories", category));

    for (auto& file : files)
      data.emplace_back(std::make_pair(std::move(file), count));

    ++count;
  }
}

Caltech101::Example Caltech101::get(size_t index) {
  auto& pair = data[index];
  auto path = datasetsimpl::join(
      root,
      "101_ObjectCategories",
      categories[size_t(pair.second)],
      pair.first);

  auto data = datasetsimpl::read_image(path, cv_transform);
  auto target = torch::from_blob(&pair.second, {1}, torch::kLong).clone();
  return {data, target};
}

torch::optional<size_t> Caltech101::size() const {
  return data.size();
}

bool Caltech101::checkIntegrity() {
  return datasetsimpl::exists(datasetsimpl::join(root, "101_ObjectCategories"));
}

} // namespace datasets
} // namespace vision
