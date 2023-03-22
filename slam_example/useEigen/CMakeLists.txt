cmake_minimum_required(VERSION 3.26)
project(useEigen)

set(CMAKE_BUILD_TYPE "Debug")

include_directories("/usr/local/include/eigen3")
add_executable(eigenMatrix eigenMatrix.cpp)