cmake_minimum_required(VERSION 2.8)
project(torchvision)
set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)

add_definitions(-D__CUDA_NO_HALF_OPERATORS__)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})

file(GLOB HEADERS torchvision/csrc/*.h)
file(GLOB CPU_HEADERS torchvision/csrc/cpu/vision_cpu.h)
file(GLOB CPU_SOURCES torchvision/csrc/cuda/*.h torchvision/csrc/cpu/*.cpp)
file(GLOB CUDA_HEADERS torchvision/csrc/cuda/vision_cuda.h)
file(GLOB CUDA_SOURCES torchvision/csrc/cuda/*.h torchvision/csrc/cuda/*.cu)
file(GLOB MODELS_HEADERS torchvision/csrc/models/*.h)
file(GLOB MODELS_SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)

add_library (${PROJECT_NAME} SHARED ${MODELS_SOURCES} ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES ${CPU_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/cpu)
install(FILES ${CUDA_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/cuda)
install(FILES ${MODELS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/models)
