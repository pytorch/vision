#ifndef CALTECH_H
#define CALTECH_H

#include <torch/torch.h>
#include "../general.h"
#include "datasetsimpl.h"

namespace vision {
namespace datasets {

class VISION_API Caltech101 : torch::data::Dataset<Caltech101> {
  using Example = torch::data::Example<>;

  std::string root;
  std::function<cv::Mat(const cv::Mat&)> cv_transform;
  std::vector<std::string> categories;
  std::vector<std::pair<std::string, long>> data;

  bool checkIntegrity();

 public:
  Caltech101(
      std::string root,
      std::function<cv::Mat(const cv::Mat&)> cv_transform =
          datasetsimpl::rgb_transform);

  Example get(size_t index) override;

  torch::optional<size_t> size() const override;
};

} // namespace datasets
} // namespace vision

#endif // CALTECH_H
