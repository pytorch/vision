cmake_minimum_required(VERSION 2.8)
project(torchvision)
set(CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)

file(GLOB VISION_HEADERS torchvision/csrc/vision.h)
file(GLOB MODELS_HEADERS torchvision/csrc/models/*.h)
file(GLOB MODELS_SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)

add_library (${PROJECT_NAME} SHARED ${MODELS_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")

add_executable(convertmodels torchvision/csrc/convert_models/convert_models.cpp)
target_link_libraries(convertmodels ${PROJECT_NAME})
target_link_libraries(convertmodels "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
install(FILES ${VISION_HEADERS} DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}")
install(FILES ${MODELS_HEADERS} DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/models")
