cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(mnist)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")


# option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
# if (DOWNLOAD_MNIST)
#   message(STATUS "Downloading MNIST dataset")
#   execute_process(
#     COMMAND python ${CMAKE_CURRENT_LIST_DIR}/tools/download_mnist.py
#       -d ${CMAKE_BINARY_DIR}/data
#     ERROR_VARIABLE DOWNLOAD_ERROR)
#   if (DOWNLOAD_ERROR)
#     message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
#   endif()
# endif()

add_executable(mnist torch.cpp)
# target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES})