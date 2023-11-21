cmake_minimum_required(VERSION 3.0)
project(test)
set(SOURCES test.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
# use opencv debug version
set(OpenCV_DIR "D:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# add opencv lib
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
message(${OpenCV_LIBS})
# set compiler flags
set(CPP_FLAGS -g)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_FLAGS}")