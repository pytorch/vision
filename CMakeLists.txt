cmake_minimum_required(VERSION 3.0)
project(torchvision)
set(CMAKE_CXX_STANDARD 14)
set(TORCHVISION_VERSION 0.6.0)

option(WITH_CUDA "Enable CUDA support" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
endif()

find_package(Python3 COMPONENTS Development)

find_package(Torch REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)


file(GLOB HEADERS torchvision/csrc/*.h)
# Image extension
file(GLOB IMAGE_HEADERS torchvision/csrc/cpu/image/*.h)
file(GLOB IMAGE_SOURCES torchvision/csrc/cpu/image/*.cpp)
file(GLOB OPERATOR_SOURCES torchvision/csrc/cpu/*.h torchvision/csrc/cpu/*.cpp ${IMAGE_HEADERS} ${IMAGE_SOURCES} ${HEADERS} torchvision/csrc/*.cpp)
if(WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} torchvision/csrc/cuda/*.h torchvision/csrc/cuda/*.cu)
endif()
file(GLOB MODELS_HEADERS torchvision/csrc/models/*.h)
file(GLOB MODELS_SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)

add_library(${PROJECT_NAME} SHARED ${MODELS_SOURCES} ${OPERATOR_SOURCES} ${IMAGE_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${PNG_LIBRARY} ${JPEG_LIBRARIES} Python3::Python)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${PNG_LIBRARY} Python3::Python)
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchVision)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${HEADERS}:${PNG_INCLUDE_DIR}:${JPEG_INCLUDE_DIRS}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TORCHVISION_CMAKECONFIG_INSTALL_DIR "share/cmake/TorchVision" CACHE STRING "install path for TorchVisionConfig.cmake")

configure_package_config_file(cmake/TorchVisionConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TorchVisionConfig.cmake"
  INSTALL_DESTINATION ${TORCHVISION_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TorchVisionConfigVersion.cmake
  VERSION ${TORCHVISION_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TorchVisionConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TorchVisionConfigVersion.cmake
  DESTINATION ${TORCHVISION_CMAKECONFIG_INSTALL_DIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT TorchVisionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT TorchVisionTargets
  NAMESPACE TorchVision::
  DESTINATION ${TORCHVISION_CMAKECONFIG_INSTALL_DIR})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES
  torchvision/csrc/cpu/vision_cpu.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cpu)
if(WITH_CUDA)
  install(FILES
    torchvision/csrc/cuda/vision_cuda.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cuda)
endif()
install(FILES ${MODELS_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/models)
