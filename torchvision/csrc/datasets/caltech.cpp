#include "caltech.h"

#include "datasetsimpl.h"

namespace vision {
namespace datasets {

Caltech101::Caltech101(const std::string& root) : root(root) {
  TORCH_CHECK(
      datasetsimpl::mkpath(root),
      "Failed to create directory \"",
      root,
      "\" for Caltech101 dataset");

  TORCH_CHECK(checkIntegrity(), "Caltech101 dataset not found or corrupted.")

  auto categories = datasetsimpl::lsdir(datasetsimpl::join(root, "caltech101"));
}

bool Caltech101::checkIntegrity() {
  return datasetsimpl::exists(datasetsimpl::join(root, "caltech101"));
}

} // namespace datasets
} // namespace vision
