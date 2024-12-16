#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

template <typename scalar_t>
at::Tensor nms_kernel_impl(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  TORCH_CHECK(dets.is_cpu(), "dets must be a CPU tensor");
  TORCH_CHECK(scores.is_cpu(), "scores must be a CPU tensor");
  TORCH_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong));

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<scalar_t>();
  auto y1 = y1_t.data_ptr<scalar_t>();
  auto x2 = x2_t.data_ptr<scalar_t>();
  auto y2 = y2_t.data_ptr<scalar_t>();
  auto areas = areas_t.data_ptr<scalar_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    keep[num_to_keep++] = i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

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


/**
 * @brief Post-processes the results of the Non-Maximum Suppression (NMS) algorithm.
 *
 * This function iterates over the boxes and determines which ones to keep based on the IOU (Intersection Over Union) keep-out mask.
 * It uses a 32-bitmask to efficiently track and suppress overlapping boxes.
 *
 * @param order A tensor containing the order of the boxes.
 * @param iou_keep_out_mask A tensor containing the IOU keep-out mask. This mask has the shape (N, N//32), where N is the number of boxes.
 * The datatype MUST be int32.
 * @param num_boxes The total number of boxes.
 * @return A tensor containing the indices of the boxes to keep.
 */

at::Tensor nms_kernel_postprocess(
    const at::Tensor& order,
    const at::Tensor& iou_keep_out_mask,
    const int64_t num_boxes) {
    // Calculate the number of 32-bit blocks needed to cover all boxes
    const int col_blocks = (num_boxes + 32 - 1) / 32;
    std::vector<unsigned long> remove_box(col_blocks);
    std::memset(&remove_box[0], 0, sizeof(unsigned long) * col_blocks);


    at::Tensor keep = at::empty({num_boxes}, order.options().dtype(at::kLong).device(at::kCPU));
    int64_t * keep_data_ptr = keep.data_ptr<int64_t>();

    unsigned long long* iou_keep_out_mask_data_ptr = (unsigned long long*)iou_keep_out_mask.data_ptr<int64_t>();
    int num_to_keep = 0;
    // Note that the iou_keep_out_mask has the shape of (N, N//32)
    // The following function iterate over each box to check if it should be kept
    for (int64_t i = 0; i < num_boxes; i++) {
      int nblock = i / 32;
      // This is equivalent to module: 31 - i % 32
      int inblock = (31 - i) & (32 -1);

      if (!(remove_box[nblock] & (1UL << inblock))){
        keep_data_ptr[num_to_keep++]=i;
        unsigned long long*p = iou_keep_out_mask_data_ptr + i*col_blocks;
        for (int j = nblock; j < col_blocks; j++){
          remove_box[j] |= p[j];
        }
      }
    }
    return order.index({keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)});
}



} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms_kernel_postprocess"), TORCH_FN(nms_kernel_postprocess));
}

} // namespace ops
} // namespace vision
