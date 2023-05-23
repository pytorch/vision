cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(test_frcnn_tracing)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

# This due to some headers importing Python.h
find_package(Python3 COMPONENTS Development)

add_executable(test_frcnn_tracing test_frcnn_tracing.cpp)
target_compile_features(test_frcnn_tracing PUBLIC cxx_range_for)
target_link_libraries(test_frcnn_tracing ${TORCH_LIBRARIES} TorchVision::TorchVision Python3::Python)
set_property(TARGET test_frcnn_tracing PROPERTY CXX_STANDARD 17)
