cmake_minimum_required(VERSION 3.26)
project(usrVisualize)

set(CMAKE_BUILD_TYPE "Debug")
#添加Eigen头文件
include_directories("/usr/local/include/eigen3")

find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(visualizeGeometry visualizeGeometry.cpp)
target_link_libraries(visualizeGeometry ${Pangolin_LIBRARIES})
