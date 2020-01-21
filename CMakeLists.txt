cmake_minimum_required(VERSION 2.8)
project(torchvision)
set(CMAKE_CXX_STANDARD 14)

option(WITH_CUDA "Enable CUDA support" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
endif()

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})

file(GLOB HEADERS torchvision/csrc/*.h)
file(GLOB OPERATOR_SOURCES torchvision/csrc/cpu/*.h torchvision/csrc/cpu/*.cpp)
if(WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} torchvision/csrc/cuda/*.h torchvision/csrc/cuda/*.cu)
endif()
file(GLOB MODELS_HEADERS torchvision/csrc/models/*.h)
file(GLOB MODELS_SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)

add_library(${PROJECT_NAME} SHARED ${MODELS_SOURCES} ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES
  torchvision/csrc/cpu/vision_cpu.h
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/cpu)
if(WITH_CUDA)
  install(FILES
    torchvision/csrc/cuda/vision_cuda.h 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/cuda)
endif()
install(FILES ${MODELS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/models)