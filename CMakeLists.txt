cmake_minimum_required(VERSION 3.1)
project(imageFilter)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} .)

add_executable(bayer bayer.cpp)
target_link_libraries(bayer ${OpenCV_LIBS})