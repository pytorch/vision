cmake_minimum_required(VERSION 2.8)
project(TorchVision)
set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE HEADERS csrc/models/*.h)
file(GLOB_RECURSE SOURCES csrc/models/*.h csrc/models/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

set(CUDNN_INCLUDE_DIR /usr/include/cuda)
find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set(INSTALL_PATH /usr/local/torchvision)

install(TARGETS ${PROJECT_NAME} DESTINATION ${INSTALL_PATH}/lib)
install(FILES ${HEADERS} DESTINATION ${INSTALL_PATH}/include/torchvision)
