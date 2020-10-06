cmake_minimum_required(VERSION 3.1)
project(torchvision)
set(CMAKE_CXX_STANDARD 14)
set(TORCHVISION_VERSION 0.7.0)

option(WITH_CUDA "Enable CUDA support" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

find_package(Python3 COMPONENTS Development)

find_package(Torch REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()

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

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
  if(WITH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819")
    foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration
      set_but_not_used field_without_dll_interface
      base_class_has_different_dll_interface
      dll_interface_conflict_none_assumed
      dll_interface_conflict_dllexport_assumed
      implicit_return_from_non_void_function
      unsigned_compare_with_zero
      declared_but_not_referenced
      bad_friend_decl)
      string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=${diag}")
    endforeach()
    CUDA_CONVERT_FLAGS(torch_cpu)
    CUDA_CONVERT_FLAGS(torch_cuda)
  endif()
endif()

add_library(${PROJECT_NAME} SHARED ${MODELS_SOURCES} ${OPERATOR_SOURCES} ${IMAGE_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${PNG_LIBRARY} ${JPEG_LIBRARIES} Python3::Python)
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchVision)

include_directories(torchvision/csrc ${JPEG_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS})
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
