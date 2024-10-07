#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TorchVision::TorchVision" for configuration ""
set_property(TARGET TorchVision::TorchVision APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(TorchVision::TorchVision PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_NOCONFIG "torch"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtorchvision.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libtorchvision.dylib"
  )

list(APPEND _cmake_import_check_targets TorchVision::TorchVision )
list(APPEND _cmake_import_check_files_for_TorchVision::TorchVision "${_IMPORT_PREFIX}/lib/libtorchvision.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
