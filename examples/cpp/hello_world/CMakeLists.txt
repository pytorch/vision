cmake_minimum_required(VERSION 3.10)
project(hello-world)

# The first thing do is to tell cmake to find the TorchVision library.
# The package pulls in all the necessary torch libraries,
# so there is no need to also add `find_package(Torch)` here.
find_package(TorchVision REQUIRED)

add_executable(hello-world main.cpp)

# We now need to link the TorchVision library to our executable.
# We can do that by using the TorchVision::TorchVision target,
# which also adds all the necessary torch dependencies.
target_link_libraries(hello-world TorchVision::TorchVision)
