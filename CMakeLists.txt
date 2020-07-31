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

file(GLOB OPERATOR_SOURCES torchvision/csrc/cpu/*.cpp torchvision/csrc/*.cpp)
file(GLOB MODELS_SOURCES torchvision/csrc/models/*.cpp)

include_directories(torchvision/cinclude/${PROJECT_NAME})

if(WITH_CUDA)
  file(GLOB OPERATOR_CUDA_SOURCES torchvision/csrc/cuda/*.cu)
  list(APPEND OPERATOR_SOURCES ${OPERATOR_CUDA_SOURCES})

  include_directories(torchvision/cinclude/${PROJECT_NAME}_cuda)
endif()

add_library(${PROJECT_NAME} SHARED ${MODELS_SOURCES} ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} Python3::Python)
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchVision)

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

install(
  DIRECTORY "torchvision/cinclude/${PROJECT_NAME}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")

if(WITH_CUDA)
  install(
    DIRECTORY "torchvision/cinclude/${PROJECT_NAME}_cuda/cuda"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
  )
endif()
