cmake_minimum_required(VERSION 3.4.1)
set(TARGET torchvision_ops)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 17)
set(LIBTORCH_HEADER_ROOT ${LIBTORCH_HEADER_ROOT})
set(LIBRARY_OUTPUT_PATH ../lib)

file(GLOB VISION_SRCS
    ../torchvision/csrc/ops/cpu/*.h
    ../torchvision/csrc/ops/cpu/*.cpp
    ../torchvision/csrc/ops/*.h
    ../torchvision/csrc/ops/*.cpp)

add_library(${TARGET} STATIC
  ${VISION_SRCS}
)

file(GLOB PYTORCH_HEADERS "${LIBTORCH_HEADER_ROOT}")
file(GLOB PYTORCH_HEADERS_CSRC "${LIBTORCH_HEADER_ROOT}/torch/csrc/api/include")
target_include_directories(${TARGET} PRIVATE
  ${PYTORCH_HEADERS}
  ${PYTORCH_HEADERS_CSRC}
)
