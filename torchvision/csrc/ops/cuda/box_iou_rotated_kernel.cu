// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Adapted from Detectron2's box_iou_rotated CUDA implementation.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "cuda_helpers.h"

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
  __host__ __device__ __forceinline__ Point(const T& px = 0, const T& py = 0)
      : x(px), y(py) {}
  __host__ __device__ __forceinline__ Point operator+(const Point& p) const {
    return Point(x + p.x, y + p.y);
  }
  __host__ __device__ __forceinline__ Point& operator+=(const Point& p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  __host__ __device__ __forceinline__ Point operator-(const Point& p) const {
    return Point(x - p.x, y - p.y);
  }
  __host__ __device__ __forceinline__ Point operator*(const T& coeff) const {
    return Point(x * coeff, y * coeff);
  }
};

template <typename T>
__host__ __device__ __forceinline__ T
dot_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.x + A.y * B.y;
}

template <typename T, typename R = T>
__host__ __device__ __forceinline__ R
cross_2d(const Point<T>& A, const Point<T>& B) {
  return static_cast<R>(A.x) * static_cast<R>(B.y) -
      static_cast<R>(B.x) * static_cast<R>(A.y);
}

template <typename T>
__host__ __device__ __forceinline__ void get_rotated_vertices(
    const RotatedBox<T>& box,
    Point<T> (&pts)[4]) {
  double theta = box.a * 0.01745329251; // M_PI / 180
  T cosTheta2 = (T)cos(theta) * 0.5f;
  T sinTheta2 = (T)sin(theta) * 0.5f;

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
__host__ __device__ __forceinline__ int get_intersection_points(
    const Point<T> (&pts1)[4],
    const Point<T> (&pts2)[4],
    Point<T> (&intersections)[24]) {
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  double EPS = 1e-5;

  int num = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      T det = cross_2d<T>(vec2[j], vec1[i]);

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

  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB > -EPS) && (APdotAD > -EPS) && (APdotAB < ABdotAB + EPS) &&
          (APdotAD < ADdotAD + EPS)) {
        intersections[num++] = pts1[i];
      }
    }
  }

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
__host__ __device__ __forceinline__ int convex_hull_graham(
    const Point<T> (&p)[24],
    const int& num_in,
    Point<T> (&q)[24],
    bool shift_to_zero = false) {
  assert(num_in >= 2);

  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto& start = p[t];

  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

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

  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }

  int k;
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2;

  for (int i = k + 1; i < num_in; i++) {
    while (m > 1) {
      auto q1 = q[i] - q[m - 2], q2 = q[m - 1] - q[m - 2];
      if (q1.x * q2.y >= q2.x * q1.y) {
        m--;
      } else {
        break;
      }
    }
    q[m++] = q[i];
  }

  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }

  return m;
}

template <typename T>
__host__ __device__ __forceinline__ T
polygon_area(const Point<T> (&q)[24], const int& m) {
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
__host__ __device__ __forceinline__ T rotated_boxes_intersection(
    const RotatedBox<T>& box1,
    const RotatedBox<T>& box2) {
  Point<T> intersectPts[24], orderedPts[24];

  Point<T> pts1[4];
  Point<T> pts2[4];
  get_rotated_vertices<T>(box1, pts1);
  get_rotated_vertices<T>(box2, pts2);

  int num = get_intersection_points<T>(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
  return polygon_area<T>(orderedPts, num_convex);
}

template <typename T>
__host__ __device__ __forceinline__ T
single_box_iou_rotated(T const* const box1_raw, T const* const box2_raw) {
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

const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 16;

template <typename T>
__global__ void box_iou_rotated_cuda_kernel(
    const int n_boxes1,
    const int n_boxes2,
    const T* dev_boxes1,
    const T* dev_boxes2,
    T* dev_ious) {
  const auto row_start = blockIdx.x * blockDim.x;
  const auto col_start = blockIdx.y * blockDim.y;

  const int row_size = min(n_boxes1 - row_start, static_cast<int>(blockDim.x));
  const int col_size = min(n_boxes2 - col_start, static_cast<int>(blockDim.y));

  __shared__ float block_boxes1[BLOCK_DIM_X * 5];
  __shared__ float block_boxes2[BLOCK_DIM_Y * 5];

  if (threadIdx.x < row_size && threadIdx.y == 0) {
    block_boxes1[threadIdx.x * 5 + 0] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 0];
    block_boxes1[threadIdx.x * 5 + 1] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 1];
    block_boxes1[threadIdx.x * 5 + 2] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 2];
    block_boxes1[threadIdx.x * 5 + 3] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 3];
    block_boxes1[threadIdx.x * 5 + 4] =
        dev_boxes1[(row_start + threadIdx.x) * 5 + 4];
  }

  if (threadIdx.x < col_size && threadIdx.y == 0) {
    block_boxes2[threadIdx.x * 5 + 0] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 0];
    block_boxes2[threadIdx.x * 5 + 1] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 1];
    block_boxes2[threadIdx.x * 5 + 2] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 2];
    block_boxes2[threadIdx.x * 5 + 3] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 3];
    block_boxes2[threadIdx.x * 5 + 4] =
        dev_boxes2[(col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size && threadIdx.y < col_size) {
    int offset = (row_start + threadIdx.x) * n_boxes2 + col_start + threadIdx.y;
    dev_ious[offset] = single_box_iou_rotated<T>(
        block_boxes1 + threadIdx.x * 5, block_boxes2 + threadIdx.y * 5);
  }
}

at::Tensor box_iou_rotated_kernel(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  TORCH_CHECK(boxes1.is_cuda(), "boxes1 must be a CUDA tensor");
  TORCH_CHECK(boxes2.is_cuda(), "boxes2 must be a CUDA tensor");
  TORCH_CHECK(
      boxes1.dim() == 2 && boxes1.size(1) == 5,
      "boxes1 should have shape (N, 5), got ",
      boxes1.sizes());
  TORCH_CHECK(
      boxes2.dim() == 2 && boxes2.size(1) == 5,
      "boxes2 should have shape (M, 5), got ",
      boxes2.sizes());
  TORCH_CHECK(
      boxes1.scalar_type() == at::kFloat, "boxes1 must be a float tensor");
  TORCH_CHECK(
      boxes2.scalar_type() == at::kFloat, "boxes2 must be a float tensor");

  at::cuda::CUDAGuard device_guard(boxes1.device());

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

  bool transpose = false;
  auto n1 = num_boxes1;
  auto n2 = num_boxes2;
  float* data1 = boxes1_contiguous.data_ptr<float>();
  float* data2 = boxes2_contiguous.data_ptr<float>();

  if (n2 > 65535 * BLOCK_DIM_Y) {
    TORCH_CHECK(
        n1 <= 65535 * BLOCK_DIM_Y, "Too many boxes for box_iou_rotated_cuda!");
    std::swap(n1, n2);
    std::swap(data1, data2);
    transpose = true;
  }

  const int blocks_x = ceil_div(static_cast<int>(n1), BLOCK_DIM_X);
  const int blocks_y = ceil_div(static_cast<int>(n2), BLOCK_DIM_Y);

  dim3 blocks(blocks_x, blocks_y);
  dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  box_iou_rotated_cuda_kernel<float><<<blocks, threads, 0, stream>>>(
      n1, n2, data1, data2, ious.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());

  if (transpose) {
    return ious.view({n1, n2}).t();
  } else {
    return ious.view({num_boxes1, num_boxes2});
  }
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::box_iou_rotated"),
      TORCH_FN(box_iou_rotated_kernel));
}

} // namespace ops
} // namespace vision
