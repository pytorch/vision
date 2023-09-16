cmake_minimum_required(VERSION 3.4.1)
set(TARGET torchvision_ops)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 17)

string(APPEND CMAKE_CXX_FLAGS " -DMOBILE")

set(build_DIR ${CMAKE_SOURCE_DIR}/build)
set(root_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

file(GLOB VISION_SRCS
    ../../torchvision/csrc/ops/cpu/*.h
    ../../torchvision/csrc/ops/cpu/*.cpp
    ../../torchvision/csrc/ops/*.h
    ../../torchvision/csrc/ops/*.cpp)

add_library(${TARGET} SHARED
  ${VISION_SRCS}
)

file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_INCLUDE_DIRS_CSRC "${build_DIR}/pytorch_android*.aar/headers/torch/csrc/api/include")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

target_compile_options(${TARGET} PRIVATE
  -fexceptions
)

set(BUILD_SUBDIR ${ANDROID_ABI})

find_library(PYTORCH_LIBRARY pytorch_jni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

find_library(FBJNI_LIBRARY fbjni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

target_include_directories(${TARGET} PRIVATE
  ${PYTORCH_INCLUDE_DIRS}
  ${PYTORCH_INCLUDE_DIRS_CSRC}
)

target_link_libraries(${TARGET} PRIVATE
  ${PYTORCH_LIBRARY}
  ${FBJNI_LIBRARY}
)
