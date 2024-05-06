cmake_minimum_required(VERSION 3.10)
project(run_model)

option(USE_TORCHVISION "Whether to link to torchvision" OFF)

find_package(Torch REQUIRED)
if(USE_TORCHVISION)
  find_package(TorchVision REQUIRED)
endif()

add_executable(run_model run_model.cpp)

target_link_libraries(run_model "${TORCH_LIBRARIES}")
if(USE_TORCHVISION)
  target_link_libraries(run_model TorchVision::TorchVision)
endif()

set_property(TARGET run_model PROPERTY CXX_STANDARD 17)
