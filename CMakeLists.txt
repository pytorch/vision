cmake_minimum_required(VERSION 2.8)
project(torchvision)
set(CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

file(GLOB HEADERS torchvision/csrc/vision.h torchvision/csrc/general.h)

file(GLOB MODELS_HEADERS torchvision/csrc/models/*.h)
file(GLOB MODELS_SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)

file(GLOB DATASETS_HEADERS torchvision/csrc/datasets/*.h)
file(GLOB DATASETS_SOURCES torchvision/csrc/datasets/*.h torchvision/csrc/datasets/*.cpp)

add_library (${PROJECT_NAME} SHARED ${MODELS_SOURCES} ${DATASETS_SOURCES})
target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

add_executable(convertmodels torchvision/csrc/convert_models/convert_models.cpp)
target_link_libraries(convertmodels "${PROJECT_NAME}")
target_link_libraries(convertmodels "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES ${MODELS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/models)
install(FILES ${DATASETS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/datasets)

