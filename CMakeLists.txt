# CMakeList.txt : CMake project for OpenCV_Matching, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("OpenCV_Matching")

# Add source to this project's executable.
add_executable (OpenCV_Matching "OpenCV_Matching.cpp" "OpenCV_Matching.h")

# TODO: Add tests and install targets if needed.
find_package(OpenCV REQUIRED)
target_link_libraries(OpenCV_Matching ${OpenCV_LIBS})