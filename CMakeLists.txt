cmake_minimum_required(VERSION 2.8)
project(torchvision)
set(CMAKE_CXX_STANDARD 11)

find_package(PNG REQUIRED)
find_package(Torch REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

file(GLOB HEADERS torchvision/csrc/vision.h)

file(GLOB MODELS_HEADERS torchvision/csrc/models/*.h)
file(GLOB MODELS_SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)

file(GLOB IMAGE_HEADERS torchvision/csrc/image/*.h)
file(GLOB IMAGE_SOURCES torchvision/csrc/image/*.h torchvision/csrc/image/*.cpp)

add_library(${PROJECT_NAME} SHARED ${MODELS_SOURCES} ${IMAGE_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC "${PNG_LIBRARY}")
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES ${MODELS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/models)
install(FILES ${IMAGE_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/image)
