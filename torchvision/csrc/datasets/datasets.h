#ifndef DATASETS_H
#define DATASETS_H

#include "caltech.h"

namespace vision {
namespace datasets {
namespace cv_transforms {
using datasetsimpl::gray_transform;
using datasetsimpl::make_transform;
using datasetsimpl::rgb_transform;
} // namespace cv_transforms
} // namespace datasets
} // namespace vision

#endif // DATASETS_H
