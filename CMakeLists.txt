cmake_minimum_required(VERSION 2.8)
project(TorchVision)
set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE HEADERS torchvision/csrc/models/*.h)
file(GLOB_RECURSE SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

set(CUDNN_INCLUDE_DIR /usr/include/cuda)
find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/torchvision)
