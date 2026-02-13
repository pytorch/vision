// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Adapted from Detectron2's box_iou_rotated implementation.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

template <typename T>
struct RotatedBox {
  T x_ctr, y_ctr, w, h, a;
};

template <typename T>
struct Point {
  T x, y;
  Point(const T& px = 0, const T& py = 0) : x(px), y(py) {}
  Point operator+(const Point& p) const {
    return Point(x + p.x, y + p.y);
  }
  Point& operator+=(const Point& p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  Point operator-(const Point& p) const {
    return Point(x - p.x, y - p.y);
  }
  Point operator*(const T& coeff) const {
    return Point(x * coeff, y * coeff);
  }
};

template <typename T>
inline T dot_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.x + A.y * B.y;
}

template <typename T>
inline T cross_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.y - B.x * A.y;
}

template <typename T>
inline void get_rotated_vertices(const RotatedBox<T>& box, Point<T> (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  double theta = box.a * 0.01745329251;
  T cosTheta2 = (T)cos(theta) * 0.5f;
  T sinTheta2 = (T)sin(theta) * 0.5f;

  // y: top --> down; x: left --> right
  pts[0].x = box.x_ctr + sinTheta2 * box.h + cosTheta2 * box.w;
  pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
  pts[1].x = box.x_ctr - sinTheta2 * box.h + cosTheta2 * box.w;
  pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
  pts[2].x = 2 * box.x_ctr - pts[0].x;
  pts[2].y = 2 * box.y_ctr - pts[0].y;
  pts[3].x = 2 * box.x_ctr - pts[1].x;
  pts[3].y = 2 * box.y_ctr - pts[1].y;
}

template <typename T>
inline int get_intersection_points(
    const Point<T> (&pts1)[4],
    const Point<T> (&pts2)[4],
    Point<T> (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // When computing the intersection area, it doesn't hurt if we have
  // more (duplicated/approximate) intersections/vertices than needed,
  // while it can cause drastic difference if we miss an intersection/vertex.
  // Therefore, we add an epsilon to relax the comparisons between
  // the float point numbers that decide the intersection points.
  double EPS = 1e-5;

  // Line test - test all line combos for intersection
  int num = 0; // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      T det = cross_2d<T>(vec2[j], vec1[i]);

      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      T t1 = cross_2d<T>(vec2[j], vec12) / det;
      T t2 = cross_2d<T>(vec1[i], vec12) / det;

      if (t1 > -EPS && t1 < 1.0f + EPS && t2 > -EPS && t2 < 1.0f + EPS) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD
      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB > -EPS) && (APdotAD > -EPS) && (APdotAB < ABdotAB + EPS) &&
          (APdotAD < ADdotAD + EPS)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto& AB = vec1[0];
    const auto& DA = vec1[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB > -EPS) && (APdotAD > -EPS) && (APdotAB < ABdotAB + EPS) &&
          (APdotAD < ADdotAD + EPS)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

template <typename T>
inline int convex_hull_graham(
    const Point<T> (&p)[24],
    const int& num_in,
    Point<T> (&q)[24],
    bool shift_to_zero = false) {
  assert(num_in >= 2);

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto& start = p[t]; // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  T dist[24];
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }

  for (int i = 1; i < num_in - 1; i++) {
    for (int j = i + 1; j < num_in; j++) {
      T crossProduct = cross_2d<T>(q[i], q[j]);
      if ((crossProduct < -1e-6) ||
          (fabs(crossProduct) < 1e-6 && dist[i] > dist[j])) {
        auto q_tmp = q[i];
        q[i] = q[j];
        q[j] = q_tmp;
        auto dist_tmp = dist[i];
        dist[i] = dist[j];
        dist[j] = dist_tmp;
      }
    }
  }

  // compute distance to origin after sort, since the points are now different.
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }

  // Step 4:
  // Make sure there are at least 2 points (that don't overlap with each other)
  // in the stack
  int k; // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2; // 2 points in the stack

  // Step 5:
  // Finally we can start the scanning process.
  // When a non-convex relationship between the 3 points is found
  // (either concave shape or duplicated points),
  // we pop the previous point from the stack
  // until the 3-point relationship is convex again, or
  // until the stack only contains two points
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1) {
      auto q1 = q[i] - q[m - 2], q2 = q[m - 1] - q[m - 2];
      // cross_2d() uses FMA and therefore computes round(round(q1.x*q2.y) -
      // q2.x*q1.y) So it may not return 0 even when q1==q2. Therefore we
      // compare round(q1.x*q2.y) and round(q2.x*q1.y) directly. (round means
      // round to nearest floating point).
      if (q1.x * q2.y >= q2.x * q1.y) {
        m--;
      } else {
        break;
      }
    }
    q[m++] = q[i];
  }

  // Step 6 (Optional):
  // In general sense we need the original coordinates, so we
  // need to shift the points back (reverting Step 2)
  // But if we're only interested in getting the area/perimeter of the shape
  // We can simply return.
  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }

  return m;
}

template <typename T>
inline T polygon_area(const Point<T> (&q)[24], const int& m) {
  if (m <= 2) {
    return 0;
  }

  T area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross_2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

template <typename T>
inline T rotated_boxes_intersection(
    const RotatedBox<T>& box1,
    const RotatedBox<T>& box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from get_intersection_points
  Point<T> intersectPts[24], orderedPts[24];

  Point<T> pts1[4];
  Point<T> pts2[4];
  get_rotated_vertices<T>(box1, pts1);
  get_rotated_vertices<T>(box2, pts2);

  int num = get_intersection_points<T>(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
  return polygon_area<T>(orderedPts, num_convex);
}

template <typename T>
inline T single_box_iou_rotated(
    T const* const box1_raw,
    T const* const box2_raw) {
  // shift center to the middle point to achieve higher precision in result
  RotatedBox<T> box1, box2;
  auto center_shift_x = (box1_raw[0] + box2_raw[0]) / 2.0;
  auto center_shift_y = (box1_raw[1] + box2_raw[1]) / 2.0;
  box1.x_ctr = box1_raw[0] - center_shift_x;
  box1.y_ctr = box1_raw[1] - center_shift_y;
  box1.w = box1_raw[2];
  box1.h = box1_raw[3];
  box1.a = box1_raw[4];
  box2.x_ctr = box2_raw[0] - center_shift_x;
  box2.y_ctr = box2_raw[1] - center_shift_y;
  box2.w = box2_raw[2];
  box2.h = box2_raw[3];
  box2.a = box2_raw[4];

  T area1 = box1.w * box1.h;
  T area2 = box2.w * box2.h;
  if (area1 < 1e-14 || area2 < 1e-14) {
    return 0.f;
  }

  T intersection = rotated_boxes_intersection<T>(box1, box2);
  T iou = intersection / (area1 + area2 - intersection);
  return iou;
}

template <typename T>
void box_iou_rotated_cpu_kernel(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    at::Tensor& ious) {
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  for (int64_t i = 0; i < num_boxes1; i++) {
    for (int64_t j = 0; j < num_boxes2; j++) {
      ious[i * num_boxes2 + j] = single_box_iou_rotated<T>(
          boxes1[i].data_ptr<T>(), boxes2[j].data_ptr<T>());
    }
  }
}

at::Tensor box_iou_rotated_kernel(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  TORCH_CHECK(boxes1.is_cpu(), "boxes1 must be a CPU tensor");
  TORCH_CHECK(boxes2.is_cpu(), "boxes2 must be a CPU tensor");
  TORCH_CHECK(
      boxes1.dim() == 2 && boxes1.size(1) == 5,
      "boxes1 should have shape (N, 5), got ",
      boxes1.sizes());
  TORCH_CHECK(
      boxes2.dim() == 2 && boxes2.size(1) == 5,
      "boxes2 should have shape (M, 5), got ",
      boxes2.sizes());
  TORCH_CHECK(
      boxes1.scalar_type() == boxes2.scalar_type(),
      "boxes1 and boxes2 must have the same dtype");

  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  if (num_boxes1 == 0 || num_boxes2 == 0) {
    return at::empty(
        {num_boxes1, num_boxes2}, boxes1.options().dtype(at::kFloat));
  }

  auto boxes1_contiguous = boxes1.contiguous();
  auto boxes2_contiguous = boxes2.contiguous();

  at::Tensor ious =
      at::empty({num_boxes1 * num_boxes2}, boxes1.options().dtype(at::kFloat));

  AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "box_iou_rotated_cpu", [&] {
    box_iou_rotated_cpu_kernel<scalar_t>(
        boxes1_contiguous, boxes2_contiguous, ious);
  });

  return ious.reshape({num_boxes1, num_boxes2});
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::box_iou_rotated"),
      TORCH_FN(box_iou_rotated_kernel));
}

} // namespace ops
} // namespace vision
