cmake_minimum_required(VERSION 3.6)
set(TARGET torchvision_ops)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)
set(LIBTORCH_HEADER_ROOT ${LIBTORCH_HEADER_ROOT})
set(LIBRARY_OUTPUT_PATH ../lib)

file(GLOB VISION_SRCS
    ../torchvision/csrc/ops/cpu/*.h
    ../torchvision/csrc/ops/cpu/*.cpp
    ../torchvision/csrc/ops/*.h
    ../torchvision/csrc/ops/*.cpp)

# Remove interpolate_aa sources as they are temporary code
# see https://github.com/pytorch/vision/pull/3761
# and using TensorIterator unavailable with iOS
# FILTER was added in CMake>=3.6 => 3.4.1 -> 3.6
list(FILTER VISION_SRCS EXCLUDE REGEX ".+(interpolate_aa).+")

add_library(${TARGET} STATIC
  ${VISION_SRCS}
)

file(GLOB PYTORCH_HEADERS "${LIBTORCH_HEADER_ROOT}")
file(GLOB PYTORCH_HEADERS_CSRC "${LIBTORCH_HEADER_ROOT}/torch/csrc/api/include")
target_include_directories(${TARGET} PRIVATE
  ${PYTORCH_HEADERS}
  ${PYTORCH_HEADERS_CSRC}
)
