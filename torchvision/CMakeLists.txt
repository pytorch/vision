cmake_minimum_required(VERSION 2.8)
project(torchvision)
set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE SOURCES cmodels/*.h cmodels/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

set(CUDNN_INCLUDE_DIR /usr/include/cuda)
find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
