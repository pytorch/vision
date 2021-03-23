#include <ATen/ATen.h>
#include <torch/library.h>

#define X1 (0)
#define Y1 (1)
#define X2 (2)
#define Y2 (3)

namespace vision {
namespace ops {

namespace {

template <typename scalar_t>
at::Tensor nms_kernel_impl(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  TORCH_CHECK(!dets.is_cuda(), "dets must be a CPU tensor");
  TORCH_CHECK(!scores.is_cuda(), "scores must be a CPU tensor");
  TORCH_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong));

  auto dets_a = dets.accessor<scalar_t, 2>();

  auto x1_t = dets.select(1, X1);
  auto y1_t = dets.select(1, Y1);
  auto x2_t = dets.select(1, X2);
  auto y2_t = dets.select(1, Y2);

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto areas = areas_t.data_ptr<scalar_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    keep[num_to_keep++] = i;
    auto ix1 = dets_a[i][X1];
    auto iy1 = dets_a[i][Y1];
    auto ix2 = dets_a[i][X2];
    auto iy2 = dets_a[i][Y2];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, dets_a[j][X1]);
      auto yy1 = std::max(iy1, dets_a[j][Y1]);
      auto xx2 = std::min(ix2, dets_a[j][X2]);
      auto yy2 = std::min(iy2, dets_a[j][Y2]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold)
        suppressed[j] = 1;
    }
  }
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_kernel", [&] {
    result = nms_kernel_impl<scalar_t>(dets, scores, iou_threshold);
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
}

} // namespace ops
} // namespace vision
