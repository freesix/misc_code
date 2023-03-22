cmake_minimum_required(VERSION 3.26)
project(useEigen)


include_directories("/usr/local/include/eigen3")
add_executable(eigenMatrix eigenMatrix.cpp)