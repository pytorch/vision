cmake_minimum_required(VERSION 2.8)
project(torchvision)
set(CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)

file(GLOB_RECURSE HEADERS torchvision/csrc/vision.h)
file(GLOB_RECURSE MODELS_HEADERS torchvision/csrc/models/*.h)
file(GLOB_RECURSE MODELS_SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)

add_library (${PROJECT_NAME} SHARED ${MODELS_SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

add_executable(convertmodels torchvision/csrc/convert_models/convert_models.cpp)
target_link_libraries(convertmodels "${PROJECT_NAME}")
target_link_libraries(convertmodels "${TORCH_LIBRARIES}")

#add_executable(testmodels test/test_models.cpp)
#target_link_libraries(testmodels "${PROJECT_NAME}")
#target_link_libraries(testmodels "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES ${MODELS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/models)
