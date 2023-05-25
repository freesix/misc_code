cmake_minimum_required(VERSION 3.26)
project(useGeometry)

set(CMAKE_BUILD_TYPE "Debug")

include_directories("/usr/local/include/eigen3")
add_executable(eigenGeometry useGeometry.cpp)