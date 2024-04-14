cmake_minimum_required(VERSION 3.26)
project(Max-Clique ${PROJECT_NAME})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)
find_package(OpenCV REQUIRED)
include_directories(include)
# message("工作路径：${WORKING_DIRECTORY}")

add_executable(${PROJECT_NAME} 
    main.cpp 
    extractFeature.cpp)   
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
