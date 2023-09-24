# cmake needs this line
cmake_minimum_required(VERSION 3.14)

# Define project name
project(opencv_example_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED) #这是根据cmake内置模块引入外部依赖包
set(CMAKE_BUILD_TYPE "debug")
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "    OpenCV library status: ${OpenCV_FOUND}") #OenCV_FOUND是否发现OpenCV的config.cmake文件
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(opencv_example example.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencv_example PRIVATE ${OpenCV_LIBS})

add_executable(cloning_gui cloning_gui.cpp)
target_link_libraries(cloning_gui ${OpenCV_LIBS})

add_executable(houghlines houghlines.cpp)
target_link_libraries(houghlines ${OpenCV_LIBS})