cmake_minimum_required(VERSION 3.18)
project(torchvision)
set(CMAKE_CXX_STANDARD 17)
file(STRINGS version.txt TORCHVISION_VERSION)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_MPS "Enable MPS support" OFF)
option(WITH_PNG "Enable features requiring LibPNG." ON)
option(WITH_JPEG "Enable features requiring LibJPEG." ON)
option(USE_PYTHON "Link to Python when building" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if(WITH_MPS)
  enable_language(OBJC OBJCXX)
  add_definitions(-DWITH_MPS)
endif()

find_package(Torch REQUIRED)

if (WITH_PNG)
    add_definitions(-DPNG_FOUND)
    find_package(PNG REQUIRED)
endif()

if (WITH_JPEG)
    add_definitions(-DJPEG_FOUND)
    find_package(JPEG REQUIRED)
endif()

if (USE_PYTHON)
  add_definitions(-DUSE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development)
endif()

function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()

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
    if(TARGET torch_cuda)
      CUDA_CONVERT_FLAGS(torch_cuda)
    endif()
    if(TARGET torch_cuda_cu)
      CUDA_CONVERT_FLAGS(torch_cuda_cu)
    endif()
    if(TARGET torch_cuda_cpp)
      CUDA_CONVERT_FLAGS(torch_cuda_cpp)
    endif()
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TVCPP torchvision/csrc)
list(APPEND ALLOW_LISTED ${TVCPP} ${TVCPP}/io/image ${TVCPP}/io/image/cpu ${TVCPP}/models ${TVCPP}/ops
  ${TVCPP}/ops/autograd ${TVCPP}/ops/cpu ${TVCPP}/io/image/cuda)
if(WITH_CUDA)
    list(APPEND ALLOW_LISTED ${TVCPP}/ops/cuda ${TVCPP}/ops/autocast)
endif()
if(WITH_MPS)
    list(APPEND ALLOW_LISTED ${TVCPP}/ops/mps)
endif()

FOREACH(DIR ${ALLOW_LISTED})
    file(GLOB ALL_SOURCES ${ALL_SOURCES} ${DIR}/*.*)
ENDFOREACH()

add_library(${PROJECT_NAME} SHARED ${ALL_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

if (WITH_PNG)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${PNG_LIBRARY})
endif()

if (WITH_JPEG)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${JPEG_LIBRARIES})
endif()

if (USE_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  EXPORT_NAME TorchVision
  INSTALL_RPATH ${TORCH_INSTALL_PREFIX}/lib)

include_directories(torchvision/csrc)

if (WITH_PNG)
    include_directories(${PNG_INCLUDE_DIRS})
endif()

if (WITH_JPEG)
    include_directories(${JPEG_INCLUDE_DIRS})
endif()

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

FOREACH(INPUT_DIR ${ALLOW_LISTED})
    string(REPLACE "${TVCPP}" "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}" OUTPUT_DIR ${INPUT_DIR})
    file(GLOB INPUT_FILES ${INPUT_DIR}/*.*)
    install(FILES ${INPUT_FILES} DESTINATION ${OUTPUT_DIR})
ENDFOREACH()
