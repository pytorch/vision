#ifndef CALTECH_H
#define CALTECH_H

#include <torch/torch.h>

namespace vision {
namespace datasets {

struct Caltech101 : torch::data::Dataset<Caltech101> {
  std::string root;

  Caltech101(const std::string& root);

  bool checkIntegrity();
};

} // namespace datasets
} // namespace vision

#endif // CALTECH_H
