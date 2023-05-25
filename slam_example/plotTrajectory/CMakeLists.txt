cmake_minimum_required(VERSION 3.26)
project(Trajectory)
set(CMAKE_BUILD_TYPE "debug")
find_package(Pangolin 0.8 REQUIRED)
include_directories("/usr/local/include/eigen3")
include_directories(${Pangolin_INCLUDE_DIRS}) #添加Pangolin的头文件(通过corePangolinxxxx.cmake)
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES}) #添加库
