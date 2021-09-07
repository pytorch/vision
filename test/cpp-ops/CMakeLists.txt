cmake_minimum_required(VERSION 3.8)
project(vision-ops-test)

set(CMAKE_CXX_STANDARD 14)

find_package(TorchVision REQUIRED)

add_executable(vision-ops-test main.cpp)
target_link_libraries(vision-ops-test TorchVision::TorchVision)
